use cuda_core::{CudaStream, DriverError};

use crate::training::runtime::Runtime;
use crate::upload::UploadedBlock;

use super::super::OptimizerTrace;
use super::super::grad_block::BlockGradBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::BlockState;
use super::adam::AdamUpdate;
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
    let mut adam = AdamUpdate::new(stream, optimizer, scratch, step, average_coefficient);
    let start = Instant::now();
    adam.update(
        &mut block.mlp_up.bias,
        &grad.d_mlp_c_fc_bias,
        &mut state.mlp_up.bias,
    )?;
    trace.adam_ms += elapsed_ms(start);

    let start = Instant::now();
    adam.update(
        &mut block.mlp_down.bias,
        &grad.d_mlp_c_proj_bias,
        &mut state.mlp_down.bias,
    )?;
    trace.adam_ms += elapsed_ms(start);
    Ok(())
}
