use std::sync::Arc;

use cuda_core::{CudaModule, DriverError, LaunchConfig};

use crate::mma::{
    NVFP4_PROJECTION_ACTIVATION_NONE, NVFP4_PROJECTION_CTA_K, NVFP4_PROJECTION_CTA_M,
    NVFP4_PROJECTION_CTA_N, NVFP4_PROJECTION_CTA_THREADS, NVFP4_PROJECTION_THREADS_PER_BLOCK,
    Nvfp4ProjectionParams, projection_cta_grid_dim, projection_cta_row_pair_tile_count,
    projection_grid_dim,
};
use crate::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;

#[path = "linear_backward/args.rs"]
mod args;
#[path = "linear_backward/bias.rs"]
mod bias;
#[path = "linear_backward/kernels.rs"]
mod kernels;
#[path = "linear_backward/ms_eden.rs"]
mod ms_eden;
pub use args::{
    LinearBackwardArgs, LinearBackwardDeviceScaleArgs, LinearBackwardInputTranspose,
    LinearBackwardMsEdenArgs, LinearBackwardMsEdenScratch, LinearBackwardMsEdenScratchBuffers,
    LinearBackwardWeightTranspose, MsEdenOperandScratch, MsEdenOperandScratchBuffer,
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
        let LinearBackwardArgs {
            stream,
            e_h,
            weight_t_h,
            e_t_h,
            input_t_h,
            dinput,
            dweight,
            dbias: _,
            token_count,
            input_dim,
            output_dim,
        } = args;

        self.backward_device_scale(LinearBackwardDeviceScaleArgs {
            stream,
            e_h,
            weight_t_h: weight_t_h.into(),
            e_t_h,
            input_t_h: input_t_h.into(),
            dinput,
            dweight,
            token_count,
            input_dim,
            output_dim,
        })
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
}

fn projection_cta_aligned(rows: u32, cols: u32, k: u32) -> bool {
    rows % NVFP4_PROJECTION_CTA_M == 0
        && cols % NVFP4_PROJECTION_CTA_N == 0
        && k % NVFP4_PROJECTION_CTA_K == 0
}
