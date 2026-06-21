use cuda_core::DriverError;

use super::blocks::run_blocks;
use super::final_head::run_final_head;
use super::types::Gpt2BackwardArgs;
use crate::GPT2_N_LAYER;
use crate::backward::{Gpt2LayerNormBackwardArgs, layer_norm_backward};
use crate::types::{Gpt2BackwardGrads, LayerNormGrads};

pub fn backward(args: Gpt2BackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    let Gpt2BackwardArgs {
        stream,
        modules,
        saved,
        weights,
        targets,
        losses,
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
        final_norm,
    } = grads;
    let LayerNormGrads {
        d_residual: _,
        d_normalized: d_final_normalized,
        d_weight: d_final_weight,
        d_bias: d_final_bias,
    } = final_norm;

    run_final_head(
        stream,
        modules,
        saved,
        weights,
        targets,
        losses,
        dlogits,
        d_final_normalized,
        d_lm_head_weight,
        scratch.final_head,
        seeds.final_head,
    )?;
    layer_norm_backward(Gpt2LayerNormBackwardArgs {
        stream,
        module: modules.final_norm,
        weights: weights.ln_f,
        saved: saved.final_norm,
        grads: LayerNormGrads {
            d_residual: blocks[GPT2_N_LAYER - 1].d_residual_out,
            d_normalized: d_final_normalized,
            d_weight: d_final_weight,
            d_bias: d_final_bias,
        },
    })?;
    run_blocks(
        stream,
        modules,
        saved,
        weights,
        &mut blocks,
        d_embedding_residual,
        &mut attention_scratch,
        &mut mlp_scratch,
        seeds,
    )
}
