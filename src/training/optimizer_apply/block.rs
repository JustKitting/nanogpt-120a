use cuda_core::{CudaStream, DriverError};

use crate::runtime::Runtime;
use crate::upload::UploadedBlock;

use super::super::grad_block::BlockGradBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::BlockState;
use super::super::optimizer_tc_scratch::AuroraScratchBuffers;
use super::layer_norm::update_layer_norm;
use super::mlp::update_mlp;
use super::qkv::update_qkv;

pub(super) fn update_block(
    stream: &CudaStream,
    runtime: &Runtime,
    block: &mut UploadedBlock,
    grad: &BlockGradBuffers,
    scratch: &mut OptimizerScratch,
    state: &mut BlockState,
    aurora: &mut AuroraScratchBuffers,
    step: u32,
) -> Result<(), DriverError> {
    let optimizer = &runtime.optimizer;
    update_layer_norm(
        stream,
        optimizer,
        &mut block.ln_1,
        &grad.ln_1,
        scratch,
        &mut state.ln_1,
        step,
    )?;
    update_qkv(stream, runtime, block, grad, scratch, state, aurora, step)?;
    update_layer_norm(
        stream,
        optimizer,
        &mut block.ln_2,
        &grad.ln_2,
        scratch,
        &mut state.ln_2,
        step,
    )?;
    update_mlp(stream, runtime, block, grad, scratch, state, aurora, step)
}
