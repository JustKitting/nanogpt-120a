use cuda_core::DriverError;

use super::types::AttentionQkvBackwardArgs;
use crate::backward::linear::{RowwiseLinearBackwardPass, run_rowwise_linear_backward};
use crate::{GPT2_FULL_ATTENTION_QKV, GPT2_N_EMBD, GPT2_QKV};

pub fn qkv_projection_backward(
    args: AttentionQkvBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    let AttentionQkvBackwardArgs {
        use_full_attention,
        stream,
        modules,
        saved,
        projections,
        d_qkv,
        d_ln_1_normalized,
        d_attn_qkv_weight,
        d_attn_qkv_bias,
        scratch,
        seeds,
    } = args;
    let output_dim = if use_full_attention {
        GPT2_FULL_ATTENTION_QKV
    } else {
        GPT2_QKV
    } as u32;

    run_rowwise_linear_backward(
        modules.linear,
        modules.quant,
        stream,
        RowwiseLinearBackwardPass {
            e: d_qkv,
            saved_input: saved.qkv_input_nvfp4,
            weight: projections.qkv_weight,
            scratch: scratch.linear,
            dinput: d_ln_1_normalized,
            dweight: d_attn_qkv_weight,
            dbias: d_attn_qkv_bias,
            row_count: saved.row_count,
            input_dim: GPT2_N_EMBD as u32,
            output_dim,
            sign_seed: seeds.sign,
            scale_seed: seeds.scale,
        },
    )
}
