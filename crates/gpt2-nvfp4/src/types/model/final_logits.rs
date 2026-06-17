use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::lm_head::{LmHeadArgs, LmHeadModule};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantModule, Nvfp4QuantRowwiseArgs};

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
    let hidden_nvfp4 = args.hidden_nvfp4;
    let mut tape = args.tape;
    let ln_f = LayerNormWeights::input_from_block(args.layer_norm_module, args.ln_f, args.hidden);
    let hidden = args.ln_f_weights.forward(ln_f)?;

    let HiddenStateDevice {
        stream,
        batch_size,
        seq_len,
        row_count,
        residual,
        normalized,
        normalized_amax,
        mean,
        inv_std,
    } = hidden;

    if let Some(tape) = tape.as_mut() {
        tape.final_norm
            .save(stream, residual, normalized, mean, inv_std)?;
    }

    args.quant_module
        .fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
            stream,
            x: normalized,
            amax: normalized_amax,
            out_fp4: &mut *hidden_nvfp4.bytes,
            out_scales: &mut *hidden_nvfp4.scales,
            out_global_scale: &mut *hidden_nvfp4.global_scales,
            group_count: row_count * crate::GPT2_N_EMBD as u32 / 16,
            row_len: crate::GPT2_N_EMBD as u32,
        })?;

    let input = Nvfp4RowwiseDeviceTensor {
        bytes: &*hidden_nvfp4.bytes,
        scales: &*hidden_nvfp4.scales,
        global_scales: &*hidden_nvfp4.global_scales,
    };
    if let Some(tape) = tape.as_mut() {
        tape.lm_head_input_nvfp4.save(stream, input)?;
    }

    args.lm_head_module.logits(LmHeadArgs {
        stream,
        input,
        weight: args.lm_head_weight,
        logits: &mut *args.logits,
        token_count: row_count,
        input_dim: crate::GPT2_N_EMBD as u32,
        vocab_size: crate::GPT2_VOCAB_SIZE as u32,
    })?;

    if let Some(tape) = tape.as_mut() {
        tape.save_logits(stream, args.logits)?;
    }

    Ok(HiddenStateDevice {
        stream,
        batch_size,
        seq_len,
        row_count,
        residual,
        normalized,
        normalized_amax,
        mean,
        inv_std,
    })
}
