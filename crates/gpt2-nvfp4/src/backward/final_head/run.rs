use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::{
    LinearBackwardInputTranspose, LinearBackwardWeightTranspose,
};
use rust_kernels_cuda::loss::{CrossEntropyArgs, LossModule};

use super::args::FinalHeadBackwardArgs;
use crate::backward::linear::{LinearBackwardCall, run_linear_backward};
use crate::{GPT2_N_EMBD, GPT2_VOCAB_SIZE};

pub fn backward(args: FinalHeadBackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    let FinalHeadBackwardArgs {
        stream,
        modules,
        logits,
        targets,
        final_normalized,
        lm_head_weight,
        losses,
        dlogits,
        d_final_normalized,
        d_lm_head_weight,
        row_count,
        scratch,
        seeds,
    } = args;

    run_loss(
        modules.loss,
        stream,
        logits,
        targets,
        losses,
        dlogits,
        &mut *scratch.linear.e_h.chunk_amax,
        row_count,
    )?;
    run_linear_backward(LinearBackwardCall {
        stream,
        module: modules.linear,
        quant: modules.quant,
        e: &*dlogits,
        weight_t: LinearBackwardWeightTranspose::Nvfp4(lm_head_weight),
        input_t: LinearBackwardInputTranspose::RowwiseNvfp4(final_normalized),
        scratch: scratch.linear,
        dinput: d_final_normalized,
        dweight: d_lm_head_weight,
        dbias: None,
        token_count: row_count,
        input_dim: GPT2_N_EMBD as u32,
        output_dim: GPT2_VOCAB_SIZE as u32,
        sign_seed: seeds.sign,
        scale_seed: seeds.scale,
        precomputed_e_amax_chunks: Some(row_count),
    })
}

fn run_loss(
    module: &LossModule,
    stream: &CudaStream,
    logits: &DeviceBuffer<f32>,
    targets: &DeviceBuffer<u32>,
    losses: &mut DeviceBuffer<f32>,
    dlogits: &mut DeviceBuffer<f32>,
    dlogits_row_amax: &mut DeviceBuffer<f32>,
    row_count: u32,
) -> Result<(), DriverError> {
    module.cross_entropy(CrossEntropyArgs {
        stream,
        logits,
        targets,
        losses,
        dlogits,
        dlogits_row_amax,
        token_count: row_count,
        vocab_size: GPT2_VOCAB_SIZE as u32,
    })
}
