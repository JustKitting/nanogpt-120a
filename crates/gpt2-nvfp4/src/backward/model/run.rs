use cuda_core::DriverError;

use super::blocks::{BlocksBackwardRun, run_blocks};
use super::final_head::run_final_head;
use super::types::Gpt2BackwardArgs;
use crate::backward::{Gpt2LayerNormBackwardArgs, layer_norm_backward};
use crate::types::Gpt2BackwardGrads;
use crate::{GPT2_EMBEDDING_DIM, GPT2_N_LAYER};
use rust_kernels_cuda::residual::ResidualGradAccumulateArgs;

pub fn backward(args: Gpt2BackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    let Gpt2BackwardArgs {
        stream,
        modules,
        saved,
        weights,
        targets,
        losses,
        extra_final_normalized_grad,
        d_lm_head_weight,
        grads,
        scratch,
        seeds,
    } = args;
    let mut attention_scratch = scratch.attention;
    let mut mlp_scratch = scratch.mlp;
    let Gpt2BackwardGrads {
        dlogits,
        d_embedding_residual,
        mut blocks,
        mut final_norm,
    } = grads;

    run_final_head(
        stream,
        modules,
        saved,
        weights,
        targets,
        losses,
        dlogits,
        &mut *final_norm.d_normalized,
        d_lm_head_weight,
        scratch.final_head,
        seeds.final_head,
    )?;
    if let Some(extra) = extra_final_normalized_grad {
        modules
            .residual
            .grad_accumulate(ResidualGradAccumulateArgs {
                stream,
                branch: extra,
                out: &mut *final_norm.d_normalized,
                len: saved.row_count * GPT2_EMBEDDING_DIM,
            })?;
    }
    layer_norm_backward(Gpt2LayerNormBackwardArgs {
        stream,
        module: modules.final_norm,
        weights: weights.ln_f,
        saved: saved.final_norm,
        grads: final_norm.reborrow_with_residual(blocks[GPT2_N_LAYER - 1].d_residual_out),
    })?;
    run_blocks(BlocksBackwardRun {
        stream,
        modules,
        saved,
        weights,
        blocks: &mut blocks,
        d_embedding_residual,
        attention_scratch: &mut attention_scratch,
        mlp_scratch: &mut mlp_scratch,
        seeds,
    })
}
