use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::{LinearBackwardModule, LinearBackwardMsEdenArgs};
use rust_kernels_cuda::loss::{CrossEntropyArgs, LossModule};

use super::args::{FinalHeadBackwardArgs, FinalHeadBackwardScratch};
use super::transpose::{decode_final_normalized, decode_lm_head_weight, transpose_dlogits};
use crate::{GPT2_CONTEXT_LEN, GPT2_N_EMBD, GPT2_VOCAB_SIZE};

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
        scratch,
        sign_seed,
        scale_seed,
    } = args;

    run_loss(modules.loss, stream, logits, targets, losses, dlogits)?;
    transpose_dlogits(
        modules.transpose,
        stream,
        &*dlogits,
        &mut *scratch.dlogits_t,
    )?;
    decode_lm_head_weight(
        modules.decode,
        stream,
        lm_head_weight,
        &mut *scratch.lm_head_weight_t,
    )?;
    decode_final_normalized(
        modules.decode,
        stream,
        final_normalized,
        &mut *scratch.final_normalized_t,
    )?;
    run_linear_backward(
        modules.linear,
        stream,
        modules.quant,
        LinearBackwardInputs {
            dlogits,
            d_final_normalized,
            d_lm_head_weight,
            scratch,
            sign_seed,
            scale_seed,
        },
    )
}

fn run_loss(
    module: &LossModule,
    stream: &CudaStream,
    logits: &DeviceBuffer<f32>,
    targets: &DeviceBuffer<u32>,
    losses: &mut DeviceBuffer<f32>,
    dlogits: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    module.cross_entropy(CrossEntropyArgs {
        stream,
        logits,
        targets,
        losses,
        dlogits,
        token_count: GPT2_CONTEXT_LEN as u32,
        vocab_size: GPT2_VOCAB_SIZE as u32,
    })
}

struct LinearBackwardInputs<'scratch, 'out> {
    dlogits: &'out mut DeviceBuffer<f32>,
    d_final_normalized: &'out mut DeviceBuffer<f32>,
    d_lm_head_weight: &'out mut DeviceBuffer<f32>,
    scratch: FinalHeadBackwardScratch<'scratch>,
    sign_seed: u32,
    scale_seed: u32,
}

fn run_linear_backward(
    module: &LinearBackwardModule,
    stream: &CudaStream,
    quant_module: &rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule,
    inputs: LinearBackwardInputs<'_, '_>,
) -> Result<(), DriverError> {
    module.backward_ms_eden(LinearBackwardMsEdenArgs {
        stream,
        quant_module,
        e: &*inputs.dlogits,
        weight_t: inputs.scratch.lm_head_weight_t,
        e_t: inputs.scratch.dlogits_t,
        input_t: inputs.scratch.final_normalized_t,
        scratch: inputs.scratch.linear,
        dinput: inputs.d_final_normalized,
        dweight: inputs.d_lm_head_weight,
        token_count: GPT2_CONTEXT_LEN as u32,
        input_dim: GPT2_N_EMBD as u32,
        output_dim: GPT2_VOCAB_SIZE as u32,
        sign_seed: inputs.sign_seed,
        scale_seed: inputs.scale_seed,
    })
}
