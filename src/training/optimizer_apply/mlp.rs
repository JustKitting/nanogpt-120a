use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD};

use crate::runtime::Runtime;
use crate::upload::UploadedBlock;

use super::super::OptimizerTrace;
use super::super::grad_block::BlockGradBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::BlockState;
use super::super::optimizer_tc_scratch::AuroraScratchBuffers;
use super::adam::update_adam_tensor;
use super::elapsed_ms;
use super::matrix::update_matrix_tensor;
use super::seed;
use std::time::Instant;

pub(super) fn update_mlp(
    stream: &CudaStream,
    runtime: &Runtime,
    block: &mut UploadedBlock,
    grad: &BlockGradBuffers,
    scratch: &mut OptimizerScratch,
    state: &mut BlockState,
    aurora: &mut AuroraScratchBuffers,
    step: u32,
    trace: &mut OptimizerTrace,
) -> Result<(), DriverError> {
    let optimizer = &runtime.optimizer;
    let start = Instant::now();
    update_matrix_tensor(
        stream,
        runtime,
        &mut block.mlp_up.weight,
        &grad.d_mlp_c_fc_weight,
        scratch,
        &mut state.mlp_up,
        aurora,
        GPT2_N_EMBD as u32,
        GPT2_MLP as u32,
        seed(step, 0x37),
        step,
    )?;
    trace.aurora_ms += elapsed_ms(start);

    let start = Instant::now();
    update_adam_tensor(
        stream,
        optimizer,
        &mut block.mlp_up.bias,
        &grad.d_mlp_c_fc_bias,
        scratch,
        &mut state.mlp_up.bias,
        step,
    )?;
    trace.adam_ms += elapsed_ms(start);

    let start = Instant::now();
    update_matrix_tensor(
        stream,
        runtime,
        &mut block.mlp_down.weight,
        &grad.d_mlp_c_proj_weight,
        scratch,
        &mut state.mlp_down,
        aurora,
        GPT2_MLP as u32,
        GPT2_N_EMBD as u32,
        seed(step, 0x41),
        step,
    )?;
    trace.aurora_ms += elapsed_ms(start);

    let start = Instant::now();
    update_adam_tensor(
        stream,
        optimizer,
        &mut block.mlp_down.bias,
        &grad.d_mlp_c_proj_bias,
        scratch,
        &mut state.mlp_down.bias,
        step,
    )?;
    trace.adam_ms += elapsed_ms(start);
    Ok(())
}
