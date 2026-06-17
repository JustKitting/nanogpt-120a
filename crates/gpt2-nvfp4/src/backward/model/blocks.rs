use cuda_core::{DeviceBuffer, DriverError};

use super::types::{Gpt2BackwardModules, Gpt2BackwardSeeds, Gpt2BackwardWeights};
use crate::backward::{
    BlockAttentionBackwardArgs, BlockAttentionBackwardScratch, BlockMlpBackwardArgs,
    MlpBackwardScratch, attention_side_backward, mlp_side_backward,
};
use crate::types::{BlockBackwardGrads, Gpt2ForwardSaved};
use crate::{GPT2_N_LAYER, backward::device_copy::copy_device};

pub(super) fn run_blocks<'a, 'scratch, 'out>(
    stream: &'a cuda_core::CudaStream,
    modules: Gpt2BackwardModules<'a>,
    saved: Gpt2ForwardSaved<'a>,
    weights: Gpt2BackwardWeights<'a>,
    blocks: &mut [BlockBackwardGrads<'out>; GPT2_N_LAYER],
    d_embedding_residual: &'out mut DeviceBuffer<f32>,
    attention_scratch: &mut BlockAttentionBackwardScratch<'scratch>,
    mlp_scratch: &mut MlpBackwardScratch<'scratch>,
    seeds: Gpt2BackwardSeeds,
) -> Result<(), DriverError> {
    for block_index in (0..GPT2_N_LAYER).rev() {
        let (lower_blocks, current_and_after) = blocks.split_at_mut(block_index);
        let current = &mut current_and_after[0];
        run_block(
            stream,
            modules,
            saved,
            weights,
            current,
            attention_scratch,
            mlp_scratch,
            seeds,
            block_index,
        )?;
        copy_block_input_gradient(
            stream,
            current,
            lower_blocks,
            d_embedding_residual,
            block_index,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_block<'a, 'scratch, 'out>(
    stream: &'a cuda_core::CudaStream,
    modules: Gpt2BackwardModules<'a>,
    saved: Gpt2ForwardSaved<'a>,
    weights: Gpt2BackwardWeights<'a>,
    current: &mut BlockBackwardGrads<'out>,
    attention_scratch: &mut BlockAttentionBackwardScratch<'scratch>,
    mlp_scratch: &mut MlpBackwardScratch<'scratch>,
    seeds: Gpt2BackwardSeeds,
    block_index: usize,
) -> Result<(), DriverError> {
    mlp_side_backward(BlockMlpBackwardArgs {
        stream,
        modules: modules.mlp,
        saved: saved.blocks[block_index],
        ln_2: weights.block_ln_2[block_index],
        mlp_projections: weights.mlp[block_index],
        grads: current.reborrow(),
        scratch: mlp_scratch.reborrow(),
        seeds: seeds.mlp[block_index],
    })?;
    attention_side_backward(BlockAttentionBackwardArgs {
        stream,
        modules: modules.attention,
        saved: saved.blocks[block_index],
        ln_1: weights.block_ln_1[block_index],
        projections: weights.attention[block_index],
        grads: current.reborrow(),
        scratch: attention_scratch.reborrow(),
        seeds: seeds.attention[block_index],
    })
}

fn copy_block_input_gradient<'out>(
    stream: &cuda_core::CudaStream,
    current: &mut BlockBackwardGrads<'out>,
    lower_blocks: &mut [BlockBackwardGrads<'out>],
    d_embedding_residual: &mut DeviceBuffer<f32>,
    block_index: usize,
) -> Result<(), DriverError> {
    if block_index == 0 {
        copy_device(stream, current.d_residual_in, d_embedding_residual)
    } else {
        copy_device(
            stream,
            current.d_residual_in,
            lower_blocks[block_index - 1].d_residual_out,
        )
    }
}
