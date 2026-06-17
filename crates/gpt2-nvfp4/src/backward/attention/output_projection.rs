use cuda_core::DriverError;

use super::linear::{AttentionLinearPass, run_attention_linear_pass};
use super::types::AttentionCProjBackwardArgs;
use crate::GPT2_N_EMBD;

pub fn c_proj_backward(args: AttentionCProjBackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    let AttentionCProjBackwardArgs {
        stream,
        modules,
        saved,
        projections,
        d_residual_after_attention,
        d_attention_out,
        d_attn_c_proj_weight,
        scratch,
        seeds,
    } = args;

    run_attention_linear_pass(
        &modules,
        stream,
        AttentionLinearPass {
            saved_input: saved.c_proj_input_nvfp4,
            weight: projections.c_proj_weight,
            scratch,
            dinput: d_attention_out,
            dweight: d_attn_c_proj_weight,
            input_dim: GPT2_N_EMBD as u32,
            output_dim: GPT2_N_EMBD as u32,
            sign_seed: seeds.sign,
            scale_seed: seeds.scale,
            e: d_residual_after_attention,
        },
    )
}
