use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel};

use crate::mma::{
    NVFP4_PROJECTION_ACTIVATION_RELU2, NVFP4_PROJECTION_THREADS_PER_BLOCK,
    Nvfp4FourSixMmaWeightTensor, Nvfp4ProjectionParams, nvfp4_projection_kernel_body,
    projection_grid_dim,
};
use crate::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};

pub struct MlpUpRelu2Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
}

pub struct MlpModule {
    module: kernels::LoadedModule,
}

impl MlpModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn up_relu2(&self, args: MlpUpRelu2Args<'_, '_>) -> Result<(), DriverError> {
        self.module.mlp_up_relu2_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: projection_grid_dim(args.token_count, args.output_dim),
                block_dim: (NVFP4_PROJECTION_THREADS_PER_BLOCK, 1, 1),
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
            Nvfp4ProjectionParams {
                token_count: args.token_count,
                input_dim: args.input_dim,
                output_dim: args.output_dim,
                weight_global_scale: args.weight.global_scale,
                bias_global_scale: args.bias.global_scale,
                residual_add: 0,
                activation: NVFP4_PROJECTION_ACTIVATION_RELU2,
            },
        )
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn mlp_up_relu2_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        mut out: DisjointSlice<f32>,
        params: Nvfp4ProjectionParams,
    ) {
        nvfp4_projection_kernel_body(
            input_bytes,
            input_scales,
            input_global_scales,
            weight_bytes,
            weight_scales,
            bias_bytes,
            bias_scales,
            &mut out,
            params,
        );
    }
}
