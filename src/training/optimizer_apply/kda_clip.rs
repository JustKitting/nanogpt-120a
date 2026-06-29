use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::{
    GPT2_FULL_ATTENTION_QKV, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_QKV, GPT2_TOKEN_ROWS,
    uses_full_attention,
};
use rust_kernels_cuda::optimizer::KdaAuroraClipArgs;
use std::time::Instant;

use crate::training::runtime::Runtime;
use crate::upload::UploadedModel;

use super::super::OptimizerTrace;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::OptimizerStateBuffers;
use super::super::tape::ForwardTapeBuffers;
use super::elapsed_ms;

const KDA_QK_CLIP_TAU: f32 = 100.0;

pub(super) fn apply_kda_aurora_clip(
    stream: &CudaStream,
    runtime: &Runtime,
    uploaded: &mut UploadedModel,
    tape: &ForwardTapeBuffers,
    scratch: &mut OptimizerScratch,
    state: &mut OptimizerStateBuffers,
    trace: &mut OptimizerTrace,
) -> Result<(), DriverError> {
    let start = Instant::now();
    for block_index in 0..uploaded.blocks.len() {
        let full_attention = uses_full_attention(block_index);
        let qkv_dim = if full_attention {
            GPT2_FULL_ATTENTION_QKV
        } else {
            GPT2_QKV
        };
        let block = &mut uploaded.blocks[block_index];
        let qkv_state = &mut state.blocks[block_index].attn_qkv.weight_aurora;
        runtime.optimizer.apply_kda_aurora_clip(KdaAuroraClipArgs {
            stream,
            qkv: tape.block_qkv(block_index),
            bytes: &mut block.attn_qkv.weight.bytes,
            scales: &mut block.attn_qkv.weight.scales,
            global_scale: &mut block.attn_qkv.weight.global_scale,
            z_master: &mut qkv_state.z_master,
            x_master: &mut qkv_state.x_master,
            momentum: &mut qkv_state.momentum,
            scores: &mut scratch.kda_clip_scores,
            amax: &mut scratch.amax,
            chunk_amax: &mut scratch.chunk_amax,
            row_count: GPT2_TOKEN_ROWS as u32,
            qkv_dim: qkv_dim as u32,
            input_dim: GPT2_N_EMBD as u32,
            embedding_dim: GPT2_N_EMBD as u32,
            head_count: GPT2_N_HEAD as u32,
            head_dim: (GPT2_N_EMBD / GPT2_N_HEAD) as u32,
            tau: KDA_QK_CLIP_TAU,
            silu_qk: (!full_attention) as u32,
        })?;
    }
    trace.kda_clip_ms += elapsed_ms(start);
    Ok(())
}
