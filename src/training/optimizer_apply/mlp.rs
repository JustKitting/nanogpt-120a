use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD};

use crate::runtime::Runtime;
use crate::upload::UploadedBlock;

use super::super::grad_block::BlockGradBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::BlockState;
use super::super::optimizer_tc_scratch::AuroraScratchBuffers;
use super::adam::update_adam_tensor;
use super::matrix::update_matrix_tensor;
use super::seed;

pub(super) fn update_mlp(
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
    update_matrix_tensor(
        stream,
        runtime,
        &mut block.mlp_up.weight,
        &grad.d_mlp_c_fc_weight,
        scratch,
        &mut state.mlp_up.weight,
        aurora,
        GPT2_N_EMBD as u32,
        GPT2_MLP as u32,
        seed(step, 0x37),
    )?;
    update_adam_tensor(
        stream,
        optimizer,
        &mut block.mlp_up.bias,
        &grad.d_mlp_c_fc_bias,
        scratch,
        &mut state.mlp_up.bias,
        step,
    )?;
    update_matrix_tensor(
        stream,
        runtime,
        &mut block.mlp_down.weight,
        &grad.d_mlp_c_proj_weight,
        scratch,
        &mut state.mlp_down.weight,
        aurora,
        GPT2_MLP as u32,
        GPT2_N_EMBD as u32,
        seed(step, 0x41),
    )?;
    update_adam_tensor(
        stream,
        optimizer,
        &mut block.mlp_down.bias,
        &grad.d_mlp_c_proj_bias,
        scratch,
        &mut state.mlp_down.bias,
        step,
    )
}
