use cuda_core::{CudaStream, DriverError};

use crate::app::runtime::Runtime;
use crate::upload::UploadedBlock;

use super::super::OptimizerTrace;
use super::super::grad_block::BlockGradBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::BlockState;
use super::adam::update_adam_tensor;
use super::elapsed_ms;
use std::time::Instant;

pub(super) fn update_mlp_biases(
    stream: &CudaStream,
    runtime: &Runtime,
    block: &mut UploadedBlock,
    grad: &BlockGradBuffers,
    scratch: &mut OptimizerScratch,
    state: &mut BlockState,
    step: u32,
    average_coefficient: f32,
    trace: &mut OptimizerTrace,
) -> Result<(), DriverError> {
    let optimizer = &runtime.optimizer;
    let start = Instant::now();
    update_adam_tensor(
        stream,
        optimizer,
        &mut block.mlp_up.bias,
        &grad.d_mlp_c_fc_bias,
        scratch,
        &mut state.mlp_up.bias,
        step,
        average_coefficient,
    )?;
    trace.adam_ms += elapsed_ms(start);

    let start = Instant::now();
    update_adam_tensor(
        stream,
        optimizer,
        &mut block.mlp_down.bias,
        &grad.d_mlp_c_proj_bias,
        scratch,
        &mut state.mlp_down.bias,
        step,
        average_coefficient,
    )?;
    trace.adam_ms += elapsed_ms(start);
    Ok(())
}
