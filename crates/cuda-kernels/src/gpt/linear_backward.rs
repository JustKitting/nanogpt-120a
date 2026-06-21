use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::mma::{
    NVFP4_PROJECTION_ACTIVATION_NONE, NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES,
    NVFP4_PROJECTION_CTA_B_PACKS, NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_K,
    NVFP4_PROJECTION_CTA_M, NVFP4_PROJECTION_CTA_N, NVFP4_PROJECTION_CTA_THREADS,
    NVFP4_PROJECTION_THREADS_PER_BLOCK, Nvfp4DeviceScaleMmaWeightTensor,
    Nvfp4FourSixMmaWeightTensor, Nvfp4ProjectionCtaTile, Nvfp4ProjectionParams,
    nvfp4_projection_cta_nobias_kernel_body, nvfp4_projection_cta_nobias_kernel_body_at_aligned,
    nvfp4_projection_nobias_kernel_body, projection_cta_grid_dim, projection_cta_tile_count,
    projection_grid_dim,
};
use crate::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use crate::nvfp4_quant::{
    MsEdenTransposeDeviceScaleQuantArgs, Nvfp4QuantModule,
    Nvfp4TransposeMsEdenDeviceScaleQuantArgs, QuartetBackwardMsEdenDeviceScaleQuantArgs,
    RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs,
};
use crate::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

#[path = "linear_backward/bias.rs"]
mod bias;
pub use bias::LINEAR_BIAS_THREADS_PER_BLOCK;

pub struct LinearBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub e_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight_t_h: Nvfp4FourSixMmaWeightTensor<'a>,
    pub e_t_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub input_t_h: Nvfp4FourSixMmaWeightTensor<'a>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: Option<&'out mut DeviceBuffer<f32>>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
}

pub struct LinearBackwardDeviceScaleArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub e_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight_t_h: Nvfp4DeviceScaleMmaWeightTensor<'a>,
    pub e_t_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub input_t_h: Nvfp4DeviceScaleMmaWeightTensor<'a>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
}

pub struct MsEdenOperandScratch<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scales: &'a mut DeviceBuffer<f32>,
    pub chunk_amax: &'a mut DeviceBuffer<f32>,
    pub global_scale: &'a mut DeviceBuffer<f32>,
}

impl<'a> MsEdenOperandScratch<'a> {
    fn rowwise(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor {
            bytes: &*self.bytes,
            scales: &*self.scales,
            global_scales: &*self.global_scales,
        }
    }

    fn device_scale_mma_weight(&self) -> Nvfp4DeviceScaleMmaWeightTensor<'_> {
        Nvfp4DeviceScaleMmaWeightTensor {
            bytes: &*self.bytes,
            scales: &*self.scales,
            global_scale: &*self.global_scale,
        }
    }
}

pub struct LinearBackwardMsEdenScratch<'a> {
    pub e_h: MsEdenOperandScratch<'a>,
    pub weight_t_h: MsEdenOperandScratch<'a>,
    pub e_t_h: MsEdenOperandScratch<'a>,
    pub input_t_h: MsEdenOperandScratch<'a>,
}

#[derive(Clone, Copy)]
pub enum LinearBackwardInputTranspose<'a> {
    Fp32(&'a DeviceBuffer<f32>),
    RowwiseNvfp4(Nvfp4RowwiseDeviceTensor<'a>),
}

#[derive(Clone, Copy)]
pub enum LinearBackwardWeightTranspose<'a> {
    Fp32(&'a DeviceBuffer<f32>),
    Nvfp4(Nvfp4DeviceTensor<'a>),
}

pub struct LinearBackwardMsEdenArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub quant_module: &'a Nvfp4QuantModule,
    pub e: &'a DeviceBuffer<f32>,
    pub weight_t: LinearBackwardWeightTranspose<'a>,
    pub input_t: LinearBackwardInputTranspose<'a>,
    pub scratch: LinearBackwardMsEdenScratch<'scratch>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: Option<&'out mut DeviceBuffer<f32>>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
    pub precomputed_e_amax_chunks: Option<u32>,
}

pub struct LinearBackwardModule {
    module: kernels::LoadedModule,
}

impl LinearBackwardModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn backward(&self, args: LinearBackwardArgs<'_, '_>) -> Result<(), DriverError> {
        let dinput_k = nvfp4_tc_matmul_padded_k(args.output_dim);
        let dweight_k = nvfp4_tc_matmul_padded_k(args.token_count);

        self.module.linear_backward_projection_device_scale_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: projection_grid_dim(args.token_count, args.input_dim),
                block_dim: (NVFP4_PROJECTION_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.e_h.bytes,
            args.e_h.scales,
            args.e_h.global_scales,
            args.weight_t_h.bytes,
            args.weight_t_h.scales,
            args.weight_t_h.global_scale,
            args.dinput,
            Nvfp4ProjectionParams {
                token_count: args.token_count,
                input_dim: dinput_k,
                output_dim: args.input_dim,
                weight_global_scale: 1.0,
                bias_global_scale: 0.0,
                residual_add: 0,
                activation: NVFP4_PROJECTION_ACTIVATION_NONE,
            },
        )?;

        self.module.linear_backward_projection_device_scale_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: projection_grid_dim(args.output_dim, args.input_dim),
                block_dim: (NVFP4_PROJECTION_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.e_t_h.bytes,
            args.e_t_h.scales,
            args.e_t_h.global_scales,
            args.input_t_h.bytes,
            args.input_t_h.scales,
            args.input_t_h.global_scale,
            args.dweight,
            Nvfp4ProjectionParams {
                token_count: args.output_dim,
                input_dim: dweight_k,
                output_dim: args.input_dim,
                weight_global_scale: 1.0,
                bias_global_scale: 0.0,
                residual_add: 0,
                activation: NVFP4_PROJECTION_ACTIVATION_NONE,
            },
        )
    }

    pub fn backward_device_scale(
        &self,
        args: LinearBackwardDeviceScaleArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let dinput_k = nvfp4_tc_matmul_padded_k(args.output_dim);
        let dweight_k = nvfp4_tc_matmul_padded_k(args.token_count);

        self.module.linear_backward_projection_device_scale_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: projection_grid_dim(args.token_count, args.input_dim),
                block_dim: (NVFP4_PROJECTION_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.e_h.bytes,
            args.e_h.scales,
            args.e_h.global_scales,
            args.weight_t_h.bytes,
            args.weight_t_h.scales,
            args.weight_t_h.global_scale,
            args.dinput,
            Nvfp4ProjectionParams {
                token_count: args.token_count,
                input_dim: dinput_k,
                output_dim: args.input_dim,
                weight_global_scale: 1.0,
                bias_global_scale: 0.0,
                residual_add: 0,
                activation: NVFP4_PROJECTION_ACTIVATION_NONE,
            },
        )?;

        self.module.linear_backward_projection_device_scale_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: projection_grid_dim(args.output_dim, args.input_dim),
                block_dim: (NVFP4_PROJECTION_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.e_t_h.bytes,
            args.e_t_h.scales,
            args.e_t_h.global_scales,
            args.input_t_h.bytes,
            args.input_t_h.scales,
            args.input_t_h.global_scale,
            args.dweight,
            Nvfp4ProjectionParams {
                token_count: args.output_dim,
                input_dim: dweight_k,
                output_dim: args.input_dim,
                weight_global_scale: 1.0,
                bias_global_scale: 0.0,
                residual_add: 0,
                activation: NVFP4_PROJECTION_ACTIVATION_NONE,
            },
        )
    }

    pub fn backward_device_scale_cta(
        &self,
        args: LinearBackwardDeviceScaleArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let dinput_k = nvfp4_tc_matmul_padded_k(args.output_dim);
        let dweight_k = nvfp4_tc_matmul_padded_k(args.token_count);
        assert_projection_cta_aligned(args.token_count, args.input_dim, dinput_k);
        assert_projection_cta_aligned(args.output_dim, args.input_dim, dweight_k);
        let dinput_grid = projection_cta_grid_dim(args.token_count, args.input_dim);
        let dweight_grid = projection_cta_grid_dim(args.output_dim, args.input_dim);
        assert!(dinput_grid.0.is_power_of_two());
        assert!(dweight_grid.0.is_power_of_two());
        let dinput_tiles = projection_cta_tile_count(args.token_count, args.input_dim);
        let dweight_tiles = projection_cta_tile_count(args.output_dim, args.input_dim);

        self.module
            .linear_backward_projection_pair_cta_device_scale_kernel(
                args.stream,
                LaunchConfig {
                    grid_dim: (dinput_tiles + dweight_tiles, 1, 1),
                    block_dim: (NVFP4_PROJECTION_CTA_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                },
                args.e_h.bytes,
                args.e_h.scales,
                args.e_h.global_scales,
                args.weight_t_h.bytes,
                args.weight_t_h.scales,
                args.weight_t_h.global_scale,
                args.dinput,
                dinput_grid.0 - 1,
                dinput_grid.0.trailing_zeros(),
                dinput_tiles,
                args.e_t_h.bytes,
                args.e_t_h.scales,
                args.e_t_h.global_scales,
                args.input_t_h.bytes,
                args.input_t_h.scales,
                args.input_t_h.global_scale,
                args.dweight,
                dweight_grid.0 - 1,
                dweight_grid.0.trailing_zeros(),
                Nvfp4ProjectionParams {
                    token_count: args.token_count,
                    input_dim: dinput_k,
                    output_dim: args.input_dim,
                    weight_global_scale: 1.0,
                    bias_global_scale: 0.0,
                    residual_add: 0,
                    activation: NVFP4_PROJECTION_ACTIVATION_NONE,
                },
                Nvfp4ProjectionParams {
                    token_count: args.output_dim,
                    input_dim: dweight_k,
                    output_dim: args.input_dim,
                    weight_global_scale: 1.0,
                    bias_global_scale: 0.0,
                    residual_add: 0,
                    activation: NVFP4_PROJECTION_ACTIVATION_NONE,
                },
            )
    }

    pub fn backward_ms_eden(
        &self,
        args: LinearBackwardMsEdenArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        if let Some(dbias) = args.dbias {
            self.module.linear_bias_grad_kernel(
                args.stream,
                LaunchConfig {
                    grid_dim: (bias::grid_dim(args.output_dim), 1, 1),
                    block_dim: (LINEAR_BIAS_THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                },
                args.e,
                dbias,
                args.token_count,
                args.output_dim,
            )?;
        }

        let scratch = args.scratch;
        let output_k = nvfp4_tc_matmul_padded_k(args.output_dim);
        let token_k = nvfp4_tc_matmul_padded_k(args.token_count);

        quantize_operand(
            args.quant_module,
            QuantizeOperandArgs {
                stream: args.stream,
                x: args.e,
                scratch: &mut *scratch.e_h.bytes,
                scales: &mut *scratch.e_h.scales,
                global_scales: &mut *scratch.e_h.global_scales,
                chunk_amax: &mut *scratch.e_h.chunk_amax,
                global_scale: &mut *scratch.e_h.global_scale,
                row_count: args.token_count,
                src_row_len: args.output_dim,
                dst_row_len: output_k,
                sign_seed: args.sign_seed,
                scale_seed: args.scale_seed,
                precomputed_chunk_count: args.precomputed_e_amax_chunks,
            },
        )?;
        match args.weight_t {
            LinearBackwardWeightTranspose::Fp32(weight_t) => {
                quantize_operand(
                    args.quant_module,
                    QuantizeOperandArgs {
                        stream: args.stream,
                        x: weight_t,
                        scratch: &mut *scratch.weight_t_h.bytes,
                        scales: &mut *scratch.weight_t_h.scales,
                        global_scales: &mut *scratch.weight_t_h.global_scales,
                        chunk_amax: &mut *scratch.weight_t_h.chunk_amax,
                        global_scale: &mut *scratch.weight_t_h.global_scale,
                        row_count: args.input_dim,
                        src_row_len: args.output_dim,
                        dst_row_len: output_k,
                        sign_seed: args.sign_seed,
                        scale_seed: args.scale_seed ^ 0x9e37_79b9,
                        precomputed_chunk_count: None,
                    },
                )?;
            }
            LinearBackwardWeightTranspose::Nvfp4(weight) => {
                args.quant_module
                    .nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
                        Nvfp4TransposeMsEdenDeviceScaleQuantArgs {
                            stream: args.stream,
                            input: weight,
                            out_fp4: &mut *scratch.weight_t_h.bytes,
                            out_scales: &mut *scratch.weight_t_h.scales,
                            out_global_scales: &mut *scratch.weight_t_h.global_scales,
                            out_chunk_amax: &mut *scratch.weight_t_h.chunk_amax,
                            out_global_scale: &mut *scratch.weight_t_h.global_scale,
                            source_rows: args.output_dim,
                            source_cols: args.input_dim,
                            dst_row_len: output_k,
                            sign_seed: args.sign_seed,
                            scale_seed: args.scale_seed ^ 0x9e37_79b9,
                        },
                    )?;
            }
        }
        quantize_transposed_operand_with_device_scale(
            args.quant_module,
            QuantizeTransposeOperandArgs {
                stream: args.stream,
                x: args.e,
                scratch: &mut *scratch.e_t_h.bytes,
                scales: &mut *scratch.e_t_h.scales,
                global_scales: &mut *scratch.e_t_h.global_scales,
                chunk_amax: &mut *scratch.e_t_h.chunk_amax,
                source_rows: args.token_count,
                source_cols: args.output_dim,
                dst_row_len: token_k,
                sign_seed: args.sign_seed,
                scale_seed: args.scale_seed ^ 0x85eb_ca6b,
            },
            &*scratch.e_h.global_scale,
        )?;
        match args.input_t {
            LinearBackwardInputTranspose::Fp32(input_t) => {
                quantize_operand(
                    args.quant_module,
                    QuantizeOperandArgs {
                        stream: args.stream,
                        x: input_t,
                        scratch: &mut *scratch.input_t_h.bytes,
                        scales: &mut *scratch.input_t_h.scales,
                        global_scales: &mut *scratch.input_t_h.global_scales,
                        chunk_amax: &mut *scratch.input_t_h.chunk_amax,
                        global_scale: &mut *scratch.input_t_h.global_scale,
                        row_count: args.input_dim,
                        src_row_len: args.token_count,
                        dst_row_len: token_k,
                        sign_seed: args.sign_seed,
                        scale_seed: args.scale_seed ^ 0xc2b2_ae35,
                        precomputed_chunk_count: None,
                    },
                )?;
            }
            LinearBackwardInputTranspose::RowwiseNvfp4(input_t) => {
                args.quant_module
                    .rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
                        RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs {
                            stream: args.stream,
                            input: input_t,
                            out_fp4: &mut *scratch.input_t_h.bytes,
                            out_scales: &mut *scratch.input_t_h.scales,
                            out_global_scales: &mut *scratch.input_t_h.global_scales,
                            out_chunk_amax: &mut *scratch.input_t_h.chunk_amax,
                            out_global_scale: &mut *scratch.input_t_h.global_scale,
                            source_rows: args.token_count,
                            source_cols: args.input_dim,
                            dst_row_len: token_k,
                            sign_seed: args.sign_seed,
                            scale_seed: args.scale_seed ^ 0xc2b2_ae35,
                        },
                    )?;
            }
        }

        self.backward_device_scale_cta(LinearBackwardDeviceScaleArgs {
            stream: args.stream,
            e_h: scratch.e_h.rowwise(),
            weight_t_h: scratch.weight_t_h.device_scale_mma_weight(),
            e_t_h: scratch.e_t_h.rowwise(),
            input_t_h: scratch.input_t_h.device_scale_mma_weight(),
            dinput: args.dinput,
            dweight: args.dweight,
            token_count: args.token_count,
            input_dim: args.input_dim,
            output_dim: args.output_dim,
        })
    }
}

fn assert_projection_cta_aligned(rows: u32, cols: u32, k: u32) {
    assert_eq!(rows % NVFP4_PROJECTION_CTA_M, 0);
    assert_eq!(cols % NVFP4_PROJECTION_CTA_N, 0);
    assert_eq!(k % NVFP4_PROJECTION_CTA_K, 0);
}

struct QuantizeOperandArgs<'a, 'out> {
    stream: &'a CudaStream,
    x: &'a DeviceBuffer<f32>,
    scratch: &'out mut DeviceBuffer<u8>,
    scales: &'out mut DeviceBuffer<u8>,
    global_scales: &'out mut DeviceBuffer<f32>,
    chunk_amax: &'out mut DeviceBuffer<f32>,
    global_scale: &'out mut DeviceBuffer<f32>,
    row_count: u32,
    src_row_len: u32,
    dst_row_len: u32,
    sign_seed: u32,
    scale_seed: u32,
    precomputed_chunk_count: Option<u32>,
}

struct QuantizeTransposeOperandArgs<'a, 'out> {
    stream: &'a CudaStream,
    x: &'a DeviceBuffer<f32>,
    scratch: &'out mut DeviceBuffer<u8>,
    scales: &'out mut DeviceBuffer<u8>,
    global_scales: &'out mut DeviceBuffer<f32>,
    chunk_amax: &'out mut DeviceBuffer<f32>,
    source_rows: u32,
    source_cols: u32,
    dst_row_len: u32,
    sign_seed: u32,
    scale_seed: u32,
}

fn quantize_operand(
    module: &Nvfp4QuantModule,
    args: QuantizeOperandArgs<'_, '_>,
) -> Result<(), DriverError> {
    if let Some(chunk_count) = args.precomputed_chunk_count {
        module.quartet_backward_ms_eden_global_scale_from_chunks(
            args.stream,
            &*args.chunk_amax,
            &mut *args.global_scale,
            chunk_count,
        )?;

        return module.fp32_to_nvfp4_ms_eden_device_scale(
            crate::nvfp4_quant::MsEdenDeviceScaleQuantArgs {
                stream: args.stream,
                x: args.x,
                out_fp4: args.scratch,
                out_scales: args.scales,
                out_global_scales: args.global_scales,
                out_chunk_amax: args.chunk_amax,
                global_scale: &*args.global_scale,
                row_count: args.row_count,
                src_row_len: args.src_row_len,
                dst_row_len: args.dst_row_len,
                scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
                sign_seed: args.sign_seed,
                scale_seed: args.scale_seed,
            },
        );
    }

    module.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale(
        QuartetBackwardMsEdenDeviceScaleQuantArgs {
            stream: args.stream,
            x: args.x,
            out_fp4: args.scratch,
            out_scales: args.scales,
            out_global_scales: args.global_scales,
            out_chunk_amax: args.chunk_amax,
            out_global_scale: args.global_scale,
            row_count: args.row_count,
            src_row_len: args.src_row_len,
            dst_row_len: args.dst_row_len,
            sign_seed: args.sign_seed,
            scale_seed: args.scale_seed,
        },
    )
}

fn quantize_transposed_operand_with_device_scale(
    module: &Nvfp4QuantModule,
    args: QuantizeTransposeOperandArgs<'_, '_>,
    global_scale: &DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    module.fp32_transpose_to_nvfp4_ms_eden_device_scale(MsEdenTransposeDeviceScaleQuantArgs {
        stream: args.stream,
        x: args.x,
        out_fp4: args.scratch,
        out_scales: args.scales,
        out_global_scales: args.global_scales,
        out_chunk_amax: args.chunk_amax,
        global_scale,
        source_rows: args.source_rows,
        source_cols: args.source_cols,
        dst_row_len: args.dst_row_len,
        scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
        sign_seed: args.sign_seed,
        scale_seed: args.scale_seed,
    })
}

#[allow(static_mut_refs)]
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn linear_backward_projection_device_scale_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut out: DisjointSlice<f32>,
        mut params: Nvfp4ProjectionParams,
    ) {
        params.weight_global_scale = weight_global_scale[0];
        nvfp4_projection_nobias_kernel_body(
            input_bytes,
            input_scales,
            input_global_scales,
            weight_bytes,
            weight_scales,
            &mut out,
            params,
        );
    }

    #[kernel]
    pub fn linear_backward_projection_cta_device_scale_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut out: DisjointSlice<f32>,
        mut params: Nvfp4ProjectionParams,
    ) {
        static mut A_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS> = SharedArray::UNINIT;
        static mut B_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS> = SharedArray::UNINIT;
        static mut A_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES> = SharedArray::UNINIT;
        static mut B_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES> = SharedArray::UNINIT;

        params.weight_global_scale = weight_global_scale[0];
        nvfp4_projection_cta_nobias_kernel_body(
            input_bytes,
            input_scales,
            input_global_scales,
            weight_bytes,
            weight_scales,
            &mut out,
            params,
            unsafe { &mut A_PACKS },
            unsafe { &mut B_PACKS },
            unsafe { &mut A_SCALES },
            unsafe { &mut B_SCALES },
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn linear_backward_projection_pair_cta_device_scale_kernel(
        dinput_input_bytes: &[u8],
        dinput_input_scales: &[u8],
        dinput_input_global_scales: &[f32],
        dinput_weight_bytes: &[u8],
        dinput_weight_scales: &[u8],
        dinput_weight_global_scale: &[f32],
        mut dinput_out: DisjointSlice<f32>,
        dinput_grid_col_mask: u32,
        dinput_grid_col_shift: u32,
        dinput_tile_count: u32,
        dweight_input_bytes: &[u8],
        dweight_input_scales: &[u8],
        dweight_input_global_scales: &[f32],
        dweight_weight_bytes: &[u8],
        dweight_weight_scales: &[u8],
        dweight_weight_global_scale: &[f32],
        mut dweight_out: DisjointSlice<f32>,
        dweight_grid_col_mask: u32,
        dweight_grid_col_shift: u32,
        mut dinput_params: Nvfp4ProjectionParams,
        mut dweight_params: Nvfp4ProjectionParams,
    ) {
        static mut A_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS> = SharedArray::UNINIT;
        static mut B_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS> = SharedArray::UNINIT;
        static mut A_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES> = SharedArray::UNINIT;
        static mut B_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES> = SharedArray::UNINIT;

        let tile_index = thread::blockIdx_x();
        let thread_id = thread::threadIdx_x();

        if tile_index < dinput_tile_count {
            let tile_col = tile_index & dinput_grid_col_mask;
            let tile_row = tile_index >> dinput_grid_col_shift;
            let tile = Nvfp4ProjectionCtaTile::from_grid_tile(tile_col, tile_row, thread_id);

            dinput_params.weight_global_scale = dinput_weight_global_scale[0];
            nvfp4_projection_cta_nobias_kernel_body_at_aligned(
                dinput_input_bytes,
                dinput_input_scales,
                dinput_input_global_scales,
                dinput_weight_bytes,
                dinput_weight_scales,
                &mut dinput_out,
                dinput_params,
                unsafe { &mut A_PACKS },
                unsafe { &mut B_PACKS },
                unsafe { &mut A_SCALES },
                unsafe { &mut B_SCALES },
                tile,
            );
        } else {
            let dweight_tile_index = tile_index - dinput_tile_count;
            let tile_col = dweight_tile_index & dweight_grid_col_mask;
            let tile_row = dweight_tile_index >> dweight_grid_col_shift;
            let tile = Nvfp4ProjectionCtaTile::from_grid_tile(tile_col, tile_row, thread_id);

            dweight_params.weight_global_scale = dweight_weight_global_scale[0];
            nvfp4_projection_cta_nobias_kernel_body_at_aligned(
                dweight_input_bytes,
                dweight_input_scales,
                dweight_input_global_scales,
                dweight_weight_bytes,
                dweight_weight_scales,
                &mut dweight_out,
                dweight_params,
                unsafe { &mut A_PACKS },
                unsafe { &mut B_PACKS },
                unsafe { &mut A_SCALES },
                unsafe { &mut B_SCALES },
                tile,
            );
        }
    }

    #[kernel]
    pub fn linear_bias_grad_kernel(
        e: &[f32],
        mut dbias: DisjointSlice<f32>,
        token_count: u32,
        output_dim: u32,
    ) {
        static mut LOCAL_SUMS: SharedArray<f32, { LINEAR_BIAS_THREADS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;
        bias::linear_bias_grad_body(e, &mut dbias, token_count, output_dim, unsafe {
            &mut LOCAL_SUMS
        });
    }
}
