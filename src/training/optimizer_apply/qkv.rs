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

pub(super) fn update_qkv_biases(
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
        &mut block.attn_qkv.bias,
        &grad.d_attn_qkv_bias,
        &mut state.attn_qkv.bias,
    )?;
    trace.adam_ms += elapsed_ms(start);

    let start = Instant::now();
    adam.update(
        &mut block.attn_c_proj.bias,
        &grad.d_attn_c_proj_bias,
        &mut state.attn_c_proj.bias,
    )?;
    trace.adam_ms += elapsed_ms(start);
    Ok(())
}
