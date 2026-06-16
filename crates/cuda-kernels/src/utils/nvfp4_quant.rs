use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, ptx_asm, thread, warp};

use crate::float_ptx::{abs_f32, max_f32};
use crate::nvfp4_cast::{e2m1_value, e4m3_value};

const THREADS_PER_BLOCK: u32 = 256;
const GROUP_SIZE_U32: u32 = 16;

#[cuda_module]
mod kernels {
    use super::*;

    const GROUP_SIZE: usize = 16;
    const FP4_MAX: f32 = 6.0;
    const FP8_MAX_FOUR_SIX: f32 = 256.0;

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_kernel(
        x: &[f32],
        amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        row_len: u32,
        scale_override: f32,
    ) {
        let lane = warp::lane_id() as usize;
        let lane_in_group = lane & 0x0f;
        let group_mask = if lane < GROUP_SIZE {
            0x0000_ffff
        } else {
            0xffff_0000
        };
        let group_leader = (lane & !0x0f) as u32;
        let groups_per_block = thread::blockDim_x() as usize / GROUP_SIZE;
        let group = thread::blockIdx_x() as usize * groups_per_block
            + thread::threadIdx_x() as usize / GROUP_SIZE;

        if group < out_scales.len() {
            let base = group * GROUP_SIZE;
            let row_len = row_len as usize;
            let scalar_scale = row_len == 0;
            let scale_row_len = if scalar_scale { usize::MAX } else { row_len };
            let row = base / scale_row_len;
            let tensor_amax = amax[row];
            let global_scale = if tensor_amax == 0.0 {
                1.0
            } else {
                tensor_amax * scale_override / (FP8_MAX_FOUR_SIX * FP4_MAX)
            };
            let writes_global_scale = if scalar_scale {
                group == 0
            } else {
                base == row * scale_row_len
            };

            unsafe {
                if writes_global_scale && lane_in_group == 0 {
                    *out_global_scale.get_unchecked_mut(row) = global_scale;
                }

                let value = x[base + lane_in_group];
                let group_amax = half_warp_max(abs_f32(value), group_mask);
                let mut scale_bits_six = 0u16;
                let mut scale_bits_four = 0u16;
                let mut scale_six = 0.0;
                let mut scale_four = 0.0;

                if lane_in_group == 0 {
                    scale_bits_six =
                        local_scale_bits(group_amax, global_scale, scale_override, 6.0);
                    scale_bits_four =
                        local_scale_bits(group_amax, global_scale, scale_override, 4.0);
                    scale_six = e4m3_value(scale_bits_six);
                    scale_four = e4m3_value(scale_bits_four);
                }

                scale_bits_six =
                    warp::shuffle_sync(group_mask, scale_bits_six as u32, group_leader) as u16;
                scale_bits_four =
                    warp::shuffle_sync(group_mask, scale_bits_four as u32, group_leader) as u16;
                scale_six = warp::shuffle_f32_sync(group_mask, scale_six, group_leader);
                scale_four = warp::shuffle_f32_sync(group_mask, scale_four, group_leader);

                let err_six =
                    half_warp_sum(candidate_error(value, scale_six, global_scale), group_mask);
                let err_four =
                    half_warp_sum(candidate_error(value, scale_four, global_scale), group_mask);
                let grid_max = if err_six <= err_four { 6.0 } else { 4.0 };
                let scale_bits = if grid_max == 6.0 {
                    scale_bits_six
                } else {
                    scale_bits_four
                };
                let scale = if grid_max == 6.0 {
                    scale_six
                } else {
                    scale_four
                };
                let scale_for_payload = if scale == 0.0 { 1.0 } else { scale };
                let inv_scale = 1.0 / (scale_for_payload * global_scale);

                if lane_in_group == 0 {
                    *out_scales.get_unchecked_mut(group) = scale_bits as u8;
                }

                if lane_in_group < GROUP_SIZE / 2 {
                    let pair = lane_in_group * 2;
                    let hi = x[base + pair] * inv_scale;
                    let lo = x[base + pair + 1] * inv_scale;
                    *out_fp4.get_unchecked_mut(base / 2 + lane_in_group) =
                        cvt_rn_satfinite_e2m1x2_f32(hi, lo);
                }
            }
        }
    }

    #[inline(always)]
    fn candidate_error(value: f32, scale: f32, global_scale: f32) -> f32 {
        let scale_for_payload = if scale == 0.0 { 1.0 } else { scale };
        let inv_scale = 1.0 / (scale_for_payload * global_scale);
        let dequant_scale = scale * global_scale;
        let packed = cvt_rn_satfinite_e2m1x2_f32(0.0, value * inv_scale);
        let dequant = e2m1_value(packed & 0x0f) * dequant_scale;
        let diff = value - dequant;
        diff * diff
    }

    #[inline(always)]
    fn half_warp_sum(mut value: f32, mask: u32) -> f32 {
        value += warp::shuffle_xor_f32_sync(mask, value, 8);
        value += warp::shuffle_xor_f32_sync(mask, value, 4);
        value += warp::shuffle_xor_f32_sync(mask, value, 2);
        value + warp::shuffle_xor_f32_sync(mask, value, 1)
    }

    #[inline(always)]
    fn half_warp_max(mut value: f32, mask: u32) -> f32 {
        value = max_f32(value, warp::shuffle_xor_f32_sync(mask, value, 8));
        value = max_f32(value, warp::shuffle_xor_f32_sync(mask, value, 4));
        value = max_f32(value, warp::shuffle_xor_f32_sync(mask, value, 2));
        max_f32(value, warp::shuffle_xor_f32_sync(mask, value, 1))
    }

    #[inline(always)]
    fn local_scale_bits(
        group_amax: f32,
        global_scale: f32,
        scale_override: f32,
        grid_max: f32,
    ) -> u16 {
        let value = group_amax * scale_override / (grid_max * global_scale);
        let packed: u16;

        unsafe {
            ptx_asm!(
                "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;",
                out("=h") packed,
                in("f") 0.0f32,
                in("f") value,
                options(register_only),
            );
        }
        packed
    }

    #[inline(always)]
    fn cvt_rn_satfinite_e2m1x2_f32(hi: f32, lo: f32) -> u8 {
        let packed: u16;
        unsafe {
            ptx_asm!(
                "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2; cvt.u16.u8 %0, tmp; }",
                out("=h") packed,
                in("f") hi,
                in("f") lo,
                options(register_only),
            );
        }
        packed as u8
    }
}

pub struct Nvfp4QuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub amax: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,
    pub out_global_scale: &'out mut DeviceBuffer<f32>,
    pub group_count: u32,
    pub scale_override: f32,
}

pub struct Nvfp4QuantRowwiseArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub amax: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,
    pub out_global_scale: &'out mut DeviceBuffer<f32>,
    pub group_count: u32,
    pub row_len: u32,
    pub scale_override: f32,
}

pub struct Nvfp4QuantModule {
    module: kernels::LoadedModule,
}

impl Nvfp4QuantModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn fp32_to_nvfp4_four_six(&self, args: Nvfp4QuantArgs<'_, '_>) -> Result<(), DriverError> {
        self.launch_fp32_to_nvfp4_four_six(
            args.stream,
            args.x,
            args.amax,
            args.out_fp4,
            args.out_scales,
            args.out_global_scale,
            args.group_count,
            0,
            args.scale_override,
        )
    }

    pub fn fp32_to_nvfp4_four_six_rowwise(
        &self,
        args: Nvfp4QuantRowwiseArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.launch_fp32_to_nvfp4_four_six(
            args.stream,
            args.x,
            args.amax,
            args.out_fp4,
            args.out_scales,
            args.out_global_scale,
            args.group_count,
            args.row_len,
            args.scale_override,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_fp32_to_nvfp4_four_six(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        amax: &DeviceBuffer<f32>,
        out_fp4: &mut DeviceBuffer<u8>,
        out_scales: &mut DeviceBuffer<u8>,
        out_global_scale: &mut DeviceBuffer<f32>,
        group_count: u32,
        row_len: u32,
        scale_override: f32,
    ) -> Result<(), DriverError> {
        let groups_per_block = THREADS_PER_BLOCK / GROUP_SIZE_U32;

        self.module.fp32_to_nvfp4_four_six_kernel(
            stream,
            LaunchConfig {
                grid_dim: (group_count.div_ceil(groups_per_block), 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            x,
            amax,
            out_fp4,
            out_scales,
            out_global_scale,
            row_len,
            scale_override,
        )
    }
}
