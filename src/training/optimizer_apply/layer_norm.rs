use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::optimizer::OptimizerModule;

use crate::upload::UploadedLayerNorm;

use super::super::grad_block::LayerNormGradBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::LayerNormState;
use super::adam::AdamUpdate;

pub(super) fn update_layer_norm(
    stream: &CudaStream,
    optimizer: &OptimizerModule,
    layer_norm: &mut UploadedLayerNorm,
    grads: &LayerNormGradBuffers,
    scratch: &mut OptimizerScratch,
    state: &mut LayerNormState,
    step: u32,
    average_coefficient: f32,
) -> Result<(), DriverError> {
    let mut adam = AdamUpdate::new(stream, optimizer, scratch, step, average_coefficient);
    adam.update(&mut layer_norm.weight, &grads.d_weight, &mut state.weight)?;
    adam.update(&mut layer_norm.bias, &grads.d_bias, &mut state.bias)
}
