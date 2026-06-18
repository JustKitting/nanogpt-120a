use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::{GPT2_N_EMBD, GPT2_QKV};

use crate::app::runtime::Runtime;
use crate::upload::UploadedBlock;

use super::super::OptimizerTrace;
use super::super::grad_block::BlockGradBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::BlockState;
use super::super::optimizer_tc_scratch::AuroraScratchBuffers;
use super::adam::update_adam_tensor;
use super::elapsed_ms;
use super::matrix::update_matrix_tensor;
use std::time::Instant;

pub(super) fn update_qkv(
    stream: &CudaStream,
    runtime: &Runtime,
    block: &mut UploadedBlock,
    grad: &BlockGradBuffers,
    scratch: &mut OptimizerScratch,
    state: &mut BlockState,
    aurora: &mut AuroraScratchBuffers,
    step: u32,
    average_coefficient: f32,
    trace: &mut OptimizerTrace,
) -> Result<(), DriverError> {
    let optimizer = &runtime.optimizer;
    let start = Instant::now();
    update_matrix_tensor(
        stream,
        runtime,
        &mut block.attn_qkv.weight,
        &grad.d_attn_qkv_weight,
        scratch,
        &mut state.attn_qkv,
        aurora,
        GPT2_N_EMBD as u32,
        GPT2_QKV as u32,
        step,
        average_coefficient,
    )?;
    trace.aurora_ms += elapsed_ms(start);

    let start = Instant::now();
    update_adam_tensor(
        stream,
        optimizer,
        &mut block.attn_qkv.bias,
        &grad.d_attn_qkv_bias,
        scratch,
        &mut state.attn_qkv.bias,
        step,
        average_coefficient,
    )?;
    trace.adam_ms += elapsed_ms(start);

    let start = Instant::now();
    update_matrix_tensor(
        stream,
        runtime,
        &mut block.attn_c_proj.weight,
        &grad.d_attn_c_proj_weight,
        scratch,
        &mut state.attn_c_proj,
        aurora,
        GPT2_N_EMBD as u32,
        GPT2_N_EMBD as u32,
        step,
        average_coefficient,
    )?;
    trace.aurora_ms += elapsed_ms(start);

    let start = Instant::now();
    update_adam_tensor(
        stream,
        optimizer,
        &mut block.attn_c_proj.bias,
        &grad.d_attn_c_proj_bias,
        scratch,
        &mut state.attn_c_proj.bias,
        step,
        average_coefficient,
    )?;
    trace.adam_ms += elapsed_ms(start);
    Ok(())
}
