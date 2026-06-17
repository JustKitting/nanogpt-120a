use cuda_core::DriverError;

use super::linear::{AttentionLinearPass, run_attention_linear_pass};
use super::types::AttentionQkvBackwardArgs;
use crate::{GPT2_N_EMBD, GPT2_QKV};

pub fn qkv_projection_backward(
    args: AttentionQkvBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    let AttentionQkvBackwardArgs {
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

    run_attention_linear_pass(
        &modules,
        stream,
        AttentionLinearPass {
            e: d_qkv,
            saved_input: saved.qkv_input_nvfp4,
            weight: projections.qkv_weight,
            scratch,
            dinput: d_ln_1_normalized,
            dweight: d_attn_qkv_weight,
            dbias: d_attn_qkv_bias,
            row_count: saved.row_count,
            input_dim: GPT2_N_EMBD as u32,
            output_dim: GPT2_QKV as u32,
            sign_seed: seeds.sign,
            scale_seed: seeds.scale,
        },
    )
}
