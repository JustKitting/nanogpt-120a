use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use super::AttentionModule;
use crate::float_ptx::fma_f32;
use crate::mma::{Nvfp4FourSixMmaWeightTensor, mma_m16n8k64_scale4x_ue4m3};
use crate::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor, nvfp4_value};

pub(crate) const ATTENTION_THREADS_PER_BLOCK: u32 = 32;
const MMA_M: u32 = 16;
const MMA_N: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct QkvProjectionParams {
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub weight_global_scale: f32,
    pub bias_global_scale: f32,
}

unsafe impl DeviceCopy for QkvProjectionParams {}

pub struct QkvProjectionArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
}

impl AttentionModule {
    pub fn qkv_projection(&self, args: QkvProjectionArgs<'_, '_>) -> Result<(), DriverError> {
        self.qkv_projection.qkv_projection_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (
                    args.output_dim.div_ceil(MMA_N),
                    args.token_count.div_ceil(MMA_M),
                    1,
                ),
                block_dim: (ATTENTION_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.weight.bytes,
            args.weight.scales,
            args.bias.bytes,
            args.bias.scales,
            args.out,
            QkvProjectionParams {
                token_count: args.token_count,
                input_dim: args.input_dim,
                output_dim: args.output_dim,
                weight_global_scale: args.weight.global_scale,
                bias_global_scale: args.bias.global_scale,
            },
        )
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    const MMA_K: u32 = 64;
    const SCALE_GROUP: u32 = 16;
    const E4M3_ONE_PACKED4: u32 = 0x3838_3838;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_projection_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        mut out: DisjointSlice<f32>,
        params: QkvProjectionParams,
    ) {
        let lane = thread::threadIdx_x();
        if lane >= ATTENTION_THREADS_PER_BLOCK {
            return;
        }

        let tile_col = thread::blockIdx_x() * MMA_N;
        let tile_row = thread::blockIdx_y() * MMA_M;
        let group = lane >> 2;
        let thread_in_group = lane & 0x3;
        let mut acc = [0.0_f32; 4];
        let mut k_base = 0;

        macro_rules! load_fragments {
            ($len:expr, $loader:ident, $($arg:expr),+ $(,)?) => {{
                let mut fragments = [0_u32; $len];
                let mut register = 0;
                while register < $len {
                    fragments[register as usize] = $loader($($arg,)* register, &params);
                    register += 1;
                }
                fragments
            }};
        }

        while k_base < params.input_dim {
            let a = load_fragments!(
                4,
                load_a_fragment,
                input_bytes,
                tile_row,
                k_base,
                group,
                thread_in_group,
            );
            let b = load_fragments!(
                2,
                load_b_fragment,
                weight_bytes,
                tile_col,
                k_base,
                group,
                thread_in_group,
            );
            let scale_a = load_a_scale4(
                input_scales,
                tile_row,
                k_base,
                group,
                thread_in_group,
                &params,
            );
            let scale_b = load_b_scale4(weight_scales, tile_col, k_base, group, &params);

            mma_m16n8k64_scale4x_ue4m3(a, b, &mut acc, scale_a, scale_b);
            k_base += MMA_K;
        }

        store_accumulator(
            acc,
            group,
            thread_in_group,
            StoreAccumulatorArgs {
                input_global_scales,
                bias_bytes,
                bias_scales,
                tile_row,
                tile_col,
                params: &params,
            },
            &mut out,
        );
    }

    #[inline(always)]
    fn load_a_fragment(
        input_bytes: &[u8],
        tile_row: u32,
        k_base: u32,
        group: u32,
        thread_in_group: u32,
        register: u32,
        params: &QkvProjectionParams,
    ) -> u32 {
        let row = tile_row + group + if register & 1 == 0 { 0 } else { 8 };
        let col = k_base + thread_in_group * 8 + if register < 2 { 0 } else { 32 };

        if row < params.token_count && col + 7 < params.input_dim {
            load_packed8(
                input_bytes,
                row as usize * params.input_dim as usize + col as usize,
            )
        } else {
            0
        }
    }

    #[inline(always)]
    fn load_b_fragment(
        weight_bytes: &[u8],
        tile_col: u32,
        k_base: u32,
        group: u32,
        thread_in_group: u32,
        register: u32,
        params: &QkvProjectionParams,
    ) -> u32 {
        let col = tile_col + group;
        let row = k_base + thread_in_group * 8 + if register == 0 { 0 } else { 32 };

        if col < params.output_dim && row + 7 < params.input_dim {
            load_packed8(
                weight_bytes,
                col as usize * params.input_dim as usize + row as usize,
            )
        } else {
            0
        }
    }

    #[inline(always)]
    fn load_a_scale4(
        input_scales: &[u8],
        tile_row: u32,
        k_base: u32,
        group: u32,
        thread_in_group: u32,
        params: &QkvProjectionParams,
    ) -> u32 {
        let row = tile_row + group + if thread_in_group == 1 { 8 } else { 0 };
        if row < params.token_count {
            let scale_base = (row * params.input_dim + k_base) / SCALE_GROUP;
            load_scale4(input_scales, scale_base as usize)
        } else {
            E4M3_ONE_PACKED4
        }
    }

    #[inline(always)]
    fn load_b_scale4(
        weight_scales: &[u8],
        tile_col: u32,
        k_base: u32,
        group: u32,
        params: &QkvProjectionParams,
    ) -> u32 {
        let col = tile_col + group;
        if col < params.output_dim {
            let scales_per_col = params.input_dim / SCALE_GROUP;
            let scale_base = col * scales_per_col + k_base / SCALE_GROUP;
            load_scale4(weight_scales, scale_base as usize)
        } else {
            E4M3_ONE_PACKED4
        }
    }

    #[inline(always)]
    fn load_packed8(bytes: &[u8], element_base: usize) -> u32 {
        let byte_base = element_base / 2;
        if byte_base + 3 < bytes.len() {
            (bytes[byte_base] as u32)
                | ((bytes[byte_base + 1] as u32) << 8)
                | ((bytes[byte_base + 2] as u32) << 16)
                | ((bytes[byte_base + 3] as u32) << 24)
        } else {
            0
        }
    }

    #[inline(always)]
    fn load_scale4(scales: &[u8], scale_base: usize) -> u32 {
        if scale_base + 3 < scales.len() {
            (scales[scale_base] as u32)
                | ((scales[scale_base + 1] as u32) << 8)
                | ((scales[scale_base + 2] as u32) << 16)
                | ((scales[scale_base + 3] as u32) << 24)
        } else {
            E4M3_ONE_PACKED4
        }
    }

    struct StoreAccumulatorArgs<'a> {
        input_global_scales: &'a [f32],
        bias_bytes: &'a [u8],
        bias_scales: &'a [u8],
        tile_row: u32,
        tile_col: u32,
        params: &'a QkvProjectionParams,
    }

    #[inline(always)]
    fn store_accumulator(
        acc: [f32; 4],
        group: u32,
        thread_in_group: u32,
        args: StoreAccumulatorArgs<'_>,
        out: &mut DisjointSlice<'_, f32>,
    ) {
        let mut i = 0;
        while i < 4 {
            let row = args.tile_row + group + if i < 2 { 0 } else { 8 };
            let col = args.tile_col + thread_in_group * 2 + (i & 1);

            if row < args.params.token_count && col < args.params.output_dim {
                let global_scale =
                    args.input_global_scales[row as usize] * args.params.weight_global_scale;
                let bias = nvfp4_value(
                    args.bias_bytes,
                    args.bias_scales,
                    args.params.bias_global_scale,
                    col as usize,
                );
                let value = fma_f32(acc[i as usize], global_scale, bias);

                unsafe {
                    *out.get_unchecked_mut(
                        row as usize * args.params.output_dim as usize + col as usize,
                    ) = value;
                }
            }

            i += 1;
        }
    }
}
