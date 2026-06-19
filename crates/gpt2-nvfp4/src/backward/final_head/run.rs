use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::{
    LinearBackwardInputTranspose, LinearBackwardModule, LinearBackwardMsEdenArgs,
    LinearBackwardWeightTranspose,
};
use rust_kernels_cuda::loss::{CrossEntropyArgs, LossModule};

use super::args::{FinalHeadBackwardArgs, FinalHeadBackwardScratch};
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
        row_count,
    )?;
    run_linear_backward(
        modules.linear,
        stream,
        modules.quant,
        LinearBackwardInputs {
            dlogits,
            final_normalized,
            lm_head_weight,
            d_final_normalized,
            d_lm_head_weight,
            scratch,
            row_count,
            sign_seed: seeds.sign,
            scale_seed: seeds.scale,
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
    row_count: u32,
) -> Result<(), DriverError> {
    module.cross_entropy(CrossEntropyArgs {
        stream,
        logits,
        targets,
        losses,
        dlogits,
        token_count: row_count,
        vocab_size: GPT2_VOCAB_SIZE as u32,
    })
}

struct LinearBackwardInputs<'scratch, 'out> {
    dlogits: &'out mut DeviceBuffer<f32>,
    final_normalized: rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor<'scratch>,
    lm_head_weight: rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor<'scratch>,
    d_final_normalized: &'out mut DeviceBuffer<f32>,
    d_lm_head_weight: &'out mut DeviceBuffer<f32>,
    scratch: FinalHeadBackwardScratch<'scratch>,
    row_count: u32,
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
        weight_t: LinearBackwardWeightTranspose::Nvfp4(inputs.lm_head_weight),
        input_t: LinearBackwardInputTranspose::RowwiseNvfp4(inputs.final_normalized),
        scratch: inputs.scratch.linear,
        dinput: inputs.d_final_normalized,
        dweight: inputs.d_lm_head_weight,
        dbias: None,
        token_count: inputs.row_count,
        input_dim: GPT2_N_EMBD as u32,
        output_dim: GPT2_VOCAB_SIZE as u32,
        sign_seed: inputs.sign_seed,
        scale_seed: inputs.scale_seed,
    })
}
