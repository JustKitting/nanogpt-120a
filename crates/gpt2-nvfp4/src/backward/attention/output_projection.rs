use cuda_core::DriverError;
use rust_kernels_cuda::linear_backward::LinearBackwardMsEdenArgs;

use super::transforms::{decode_rowwise_t, decode_weight_t, transpose_f32};
use super::types::{AttentionCProjBackwardArgs, AttentionCProjScratch};
use crate::{GPT2_CONTEXT_LEN, GPT2_N_EMBD};

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
    let AttentionCProjScratch {
        error_t,
        weight_t,
        input_t,
        linear,
    } = scratch;

    decode_weight_t(
        modules.decode,
        stream,
        projections.c_proj_weight,
        weight_t,
        GPT2_N_EMBD,
        GPT2_N_EMBD,
    )?;
    transpose_f32(
        modules.transpose,
        stream,
        d_residual_after_attention,
        error_t,
        GPT2_CONTEXT_LEN,
        GPT2_N_EMBD,
    )?;
    decode_rowwise_t(
        modules.decode,
        stream,
        saved.c_proj_input_nvfp4,
        input_t,
        GPT2_CONTEXT_LEN,
        GPT2_N_EMBD,
    )?;

    modules.linear.backward_ms_eden(LinearBackwardMsEdenArgs {
        stream,
        quant_module: modules.quant,
        e: d_residual_after_attention,
        weight_t,
        e_t: error_t,
        input_t,
        scratch: linear,
        dinput: d_attention_out,
        dweight: d_attn_c_proj_weight,
        token_count: GPT2_CONTEXT_LEN as u32,
        input_dim: GPT2_N_EMBD as u32,
        output_dim: GPT2_N_EMBD as u32,
        sign_seed: seeds.sign,
        scale_seed: seeds.scale,
    })
}
