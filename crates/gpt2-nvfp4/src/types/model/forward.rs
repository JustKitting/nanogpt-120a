use cuda_core::DriverError;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;

use super::args::Gpt2ForwardArgs;
use super::final_logits::{FinalForwardArgs, finish_forward};
use super::weights::Gpt2Weights;
use crate::types::{AttentionProjectionTensors, BlockForwardArgs, HiddenStateDevice};
use crate::uses_full_attention;

pub(super) fn forward<'a>(
    weights: &Gpt2Weights,
    args: Gpt2ForwardArgs<'a>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let Gpt2ForwardArgs {
        embeddings,
        attention_module,
        attention_tc_module,
        quant_module,
        layer_norm_module,
        mlp_module,
        lm_head_module,
        mut hidden_nvfp4,
        mut attention_tc_scratch,
        mut mlp_activation_nvfp4,
        attention_qkv_weights,
        attention_qkv_biases,
        attention_c_proj_weights,
        attention_c_proj_biases,
        block_ln_1,
        block_ln_2,
        mlp_up,
        mlp_down,
        ln_f,
        attention_qkv,
        attention_log_sum_exp,
        mlp_pre_activation,
        mlp_activation,
        logits,
        mut tape,
    } = args;

    let lm_head_weight = Nvfp4FourSixMmaWeightTensor {
        bytes: embeddings.token_embedding.bytes,
        scales: embeddings.token_embedding.scales,
        global_scale: embeddings.token_embedding.global_scale,
    };

    let mut hidden = weights.embeddings.forward(embeddings)?;

    for (block_index, block) in weights.h.iter().enumerate() {
        hidden = block.forward(BlockForwardArgs {
            use_full_attention: uses_full_attention(block_index),
            attention_module,
            attention_tc_module,
            quant_module,
            layer_norm_module,
            mlp_module,
            hidden_nvfp4: hidden_nvfp4.reborrow(),
            attention_tc_scratch: attention_tc_scratch.reborrow(),
            mlp_activation_nvfp4: mlp_activation_nvfp4.reborrow(),
            projections: AttentionProjectionTensors {
                qkv_weight: attention_qkv_weights[block_index],
                qkv_bias: attention_qkv_biases[block_index],
                c_proj_weight: attention_c_proj_weights[block_index],
                c_proj_bias: attention_c_proj_biases[block_index],
            },
            ln_1: block_ln_1[block_index],
            ln_2: block_ln_2[block_index],
            mlp_up: mlp_up[block_index],
            mlp_down: mlp_down[block_index],
            qkv: &mut *attention_qkv,
            attention_log_sum_exp: &mut *attention_log_sum_exp,
            mlp_pre_activation: &mut *mlp_pre_activation,
            mlp_activation: &mut *mlp_activation,
            hidden,
            tape: tape.as_mut().map(|tape| tape.block(block_index)),
        })?;
    }

    finish_forward(FinalForwardArgs {
        ln_f_weights: &weights.ln_f,
        layer_norm_module,
        quant_module,
        lm_head_module,
        ln_f,
        hidden_nvfp4,
        lm_head_weight,
        logits,
        tape,
        hidden,
    })
}
