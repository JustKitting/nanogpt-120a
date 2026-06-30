use cuda_core::DriverError;

use super::types::BlockAttentionBackwardArgs;
use crate::backward::residual::residual_grad_add;
use crate::backward::{
    AttentionCProjBackwardArgs, AttentionCoreBackwardArgs, AttentionQkvBackwardArgs,
    Gpt2LayerNormBackwardArgs, attention_c_proj_backward, causal_attention_backward,
    layer_norm_backward, qkv_projection_backward,
};
use crate::types::{BlockBackwardGrads, LayerNormGrads};

pub fn attention_side_backward(
    args: BlockAttentionBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    let BlockAttentionBackwardArgs {
        use_full_attention,
        stream,
        modules,
        saved,
        ln_1,
        projections,
        grads,
        scratch,
        seeds,
    } = args;
    let BlockBackwardGrads {
        d_residual_in,
        ln_1: ln_1_grads,
        d_qkv,
        d_attention_out,
        d_residual_after_attention,
        d_attn_qkv_weight,
        d_attn_qkv_bias,
        d_attn_c_proj_weight,
        d_attn_c_proj_bias,
        ..
    } = grads;
    let LayerNormGrads {
        d_residual: d_ln_1_residual,
        d_normalized: d_ln_1_normalized,
        d_weight: d_ln_1_weight,
        d_bias: d_ln_1_bias,
    } = ln_1_grads;

    attention_c_proj_backward(AttentionCProjBackwardArgs {
        stream,
        modules: modules.linear,
        saved,
        projections,
        d_residual_after_attention: &*d_residual_after_attention,
        d_attention_out,
        d_attn_c_proj_weight,
        d_attn_c_proj_bias,
        scratch: scratch.c_proj,
        seeds: seeds.c_proj,
    })?;
    causal_attention_backward(AttentionCoreBackwardArgs {
        use_full_attention,
        stream,
        module: modules.attention,
        tc_module: modules.f16_tc,
        saved,
        d_attention_out: &*d_attention_out,
        d_qkv,
        scratch: scratch.core,
    })?;
    qkv_projection_backward(AttentionQkvBackwardArgs {
        use_full_attention,
        stream,
        modules: modules.linear,
        saved,
        projections,
        d_qkv: &*d_qkv,
        d_ln_1_normalized,
        d_attn_qkv_weight,
        d_attn_qkv_bias,
        scratch: scratch.qkv,
        seeds: seeds.qkv,
    })?;
    layer_norm_backward(Gpt2LayerNormBackwardArgs {
        stream,
        module: modules.layer_norm,
        weights: ln_1,
        saved: saved.ln_1,
        grads: LayerNormGrads {
            d_residual: d_ln_1_residual,
            d_normalized: d_ln_1_normalized,
            d_weight: d_ln_1_weight,
            d_bias: d_ln_1_bias,
        },
    })?;

    residual_grad_add(
        modules.residual,
        stream,
        &*d_residual_after_attention,
        &*d_ln_1_residual,
        d_residual_in,
        saved.row_count,
    )
}
