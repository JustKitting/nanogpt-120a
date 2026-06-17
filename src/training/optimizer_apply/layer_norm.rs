use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::optimizer::OptimizerModule;

use crate::upload::UploadedLayerNorm;

use super::super::grad_block::LayerNormGradBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::LayerNormState;
use super::adam::update_adam_tensor;

pub(super) fn update_layer_norm(
    stream: &CudaStream,
    optimizer: &OptimizerModule,
    layer_norm: &mut UploadedLayerNorm,
    grads: &LayerNormGradBuffers,
    scratch: &mut OptimizerScratch,
    state: &mut LayerNormState,
    step: u32,
) -> Result<(), DriverError> {
    update_adam_tensor(
        stream,
        optimizer,
        &mut layer_norm.weight,
        &grads.d_weight,
        scratch,
        &mut state.weight,
        step,
    )?;
    update_adam_tensor(
        stream,
        optimizer,
        &mut layer_norm.bias,
        &grads.d_bias,
        scratch,
        &mut state.bias,
        step,
    )
}
