use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel};

use crate::mma::{
    NVFP4_PROJECTION_ACTIVATION_NONE, NVFP4_PROJECTION_THREADS_PER_BLOCK,
    Nvfp4FourSixMmaWeightTensor, Nvfp4ProjectionParams, nvfp4_projection_nobias_kernel_body,
    projection_grid_dim,
};
use crate::nvfp4::Nvfp4RowwiseDeviceTensor;

pub struct LinearBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub e_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight_t_h: Nvfp4FourSixMmaWeightTensor<'a>,
    pub e_t_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub input_t_h: Nvfp4FourSixMmaWeightTensor<'a>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
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
        self.module.linear_backward_projection_kernel(
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
            args.dinput,
            Nvfp4ProjectionParams {
                token_count: args.token_count,
                input_dim: args.output_dim,
                output_dim: args.input_dim,
                weight_global_scale: args.weight_t_h.global_scale,
                bias_global_scale: 0.0,
                residual_add: 0,
                activation: NVFP4_PROJECTION_ACTIVATION_NONE,
            },
        )?;

        self.module.linear_backward_projection_kernel(
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
            args.dweight,
            Nvfp4ProjectionParams {
                token_count: args.output_dim,
                input_dim: args.token_count,
                output_dim: args.input_dim,
                weight_global_scale: args.input_t_h.global_scale,
                bias_global_scale: 0.0,
                residual_add: 0,
                activation: NVFP4_PROJECTION_ACTIVATION_NONE,
            },
        )
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn linear_backward_projection_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        mut out: DisjointSlice<f32>,
        params: Nvfp4ProjectionParams,
    ) {
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
}
