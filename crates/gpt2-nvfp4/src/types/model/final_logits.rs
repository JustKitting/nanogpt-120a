use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::lm_head::{LmHeadArgs, LmHeadModule};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use crate::types::{
    Gpt2ForwardTape, HiddenStateDevice, HiddenStateNvfp4, LayerNormTensors, LayerNormWeights,
};

pub(super) struct FinalForwardArgs<'a, 'w> {
    pub ln_f_weights: &'w LayerNormWeights,
    pub layer_norm_module: &'a LayerNormModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub lm_head_module: &'a LmHeadModule,
    pub ln_f: LayerNormTensors<'a>,
    pub hidden_nvfp4: HiddenStateNvfp4<'a>,
    pub lm_head_weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub logits: &'a mut DeviceBuffer<f32>,
    pub tape: Option<Gpt2ForwardTape<'a>>,
    pub hidden: HiddenStateDevice<'a>,
}

pub(super) fn finish_forward<'a>(
    args: FinalForwardArgs<'a, '_>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let mut hidden_nvfp4 = args.hidden_nvfp4;
    let mut tape = args.tape;
    let ln_f = LayerNormWeights::input_from_block(args.layer_norm_module, args.ln_f, args.hidden);
    let hidden = args
        .ln_f_weights
        .forward_with_tape(ln_f, tape.as_mut().map(|tape| &mut tape.final_norm))?;

    hidden_nvfp4.quantize_precomputed_amax(
        args.quant_module,
        hidden.stream,
        &*hidden.normalized,
        &*hidden.normalized_amax,
        hidden.row_count,
        crate::GPT2_EMBEDDING_DIM,
    )?;

    let input = hidden_nvfp4.device();
    if let Some(tape) = tape.as_mut() {
        tape.lm_head_input_nvfp4.save(hidden.stream, input)?;
    }

    args.lm_head_module.logits(LmHeadArgs {
        stream: hidden.stream,
        input,
        weight: args.lm_head_weight,
        logits: &mut *args.logits,
        token_count: hidden.row_count,
        input_dim: crate::GPT2_EMBEDDING_DIM,
        vocab_size: crate::GPT2_VOCAB_DIM,
    })?;

    Ok(hidden)
}
