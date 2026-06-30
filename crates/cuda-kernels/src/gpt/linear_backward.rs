use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use crate::mma::{
    NVFP4_PROJECTION_ACTIVATION_NONE, NVFP4_PROJECTION_CTA_K, NVFP4_PROJECTION_CTA_M,
    NVFP4_PROJECTION_CTA_N, NVFP4_PROJECTION_CTA_THREADS, NVFP4_PROJECTION_THREADS_PER_BLOCK,
    Nvfp4ProjectionParams, projection_cta_grid_dim, projection_cta_row_pair_tile_count,
    projection_grid_dim,
};
use crate::nvfp4_quant::{
    MsEdenPairDeviceScaleQuantArgs, Nvfp4QuantModule, Nvfp4TransposeMsEdenDeviceScaleQuantArgs,
    QuartetBackwardMsEdenDeviceScaleQuantArgs, RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs,
};
use crate::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

#[path = "linear_backward/args.rs"]
mod args;
#[path = "linear_backward/bias.rs"]
mod bias;
#[path = "linear_backward/kernels.rs"]
mod kernels;
pub use args::{
    LinearBackwardArgs, LinearBackwardDeviceScaleArgs, LinearBackwardInputTranspose,
    LinearBackwardMsEdenArgs, LinearBackwardMsEdenScratch, LinearBackwardWeightTranspose,
    MsEdenOperandScratch,
};
pub use bias::LINEAR_BIAS_THREADS_PER_BLOCK;

pub struct LinearBackwardModule {
    module: kernels::module::LoadedModule,
}

impl LinearBackwardModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::module::from_module(module)?,
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
        if !projection_cta_aligned(args.token_count, args.input_dim, dinput_k)
            || !projection_cta_aligned(args.output_dim, args.input_dim, dweight_k)
        {
            return self.backward_device_scale(args);
        }
        let dinput_grid = projection_cta_grid_dim(args.token_count, args.input_dim);
        let dweight_grid = projection_cta_grid_dim(args.output_dim, args.input_dim);
        assert!(dinput_grid.0.is_power_of_two());
        assert!(dweight_grid.0.is_power_of_two());
        let dinput_tiles = projection_cta_row_pair_tile_count(args.token_count, args.input_dim);
        let dweight_tiles = projection_cta_row_pair_tile_count(args.output_dim, args.input_dim);

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

        args.quant_module
            .fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
                MsEdenPairDeviceScaleQuantArgs {
                    stream: args.stream,
                    x: args.e,
                    out_fp4: &mut *scratch.e_h.bytes,
                    out_scales: &mut *scratch.e_h.scales,
                    out_global_scales: &mut *scratch.e_h.global_scales,
                    transpose_out_fp4: &mut *scratch.e_t_h.bytes,
                    transpose_out_scales: &mut *scratch.e_t_h.scales,
                    transpose_out_global_scales: &mut *scratch.e_t_h.global_scales,
                    out_chunk_amax: &mut *scratch.e_h.chunk_amax,
                    out_global_scale: &mut *scratch.e_h.global_scale,
                    row_count: args.token_count,
                    src_row_len: args.output_dim,
                    dst_row_len: output_k,
                    transpose_dst_row_len: token_k,
                    scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
                    sign_seed: args.sign_seed,
                    scale_seed: args.scale_seed,
                    transpose_scale_seed: args.scale_seed ^ 0x85eb_ca6b,
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
                    .nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
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
                    .rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
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

fn projection_cta_aligned(rows: u32, cols: u32, k: u32) -> bool {
    rows % NVFP4_PROJECTION_CTA_M == 0
        && cols % NVFP4_PROJECTION_CTA_N == 0
        && k % NVFP4_PROJECTION_CTA_K == 0
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

        return module.fp32_to_nvfp4_ms_eden_device_scale_no_chunk_amax(
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

    module.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
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
