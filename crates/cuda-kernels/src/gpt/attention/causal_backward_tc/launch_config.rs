use cuda_core::LaunchConfig;

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::kda_launch::{KDA_CHUNK_SIZE, KDA_DECAY_SCALE};

const SOFTMAX_D_THREADS_PER_BLOCK: u32 = 64;

pub(super) fn tc_params(
    row_count: u32,
    seq_len: u32,
    batch_size: u32,
    embedding_dim: u32,
    qkv_dim: u32,
    head_count: u32,
    head_dim: u32,
) -> CausalAttentionParams {
    CausalAttentionParams {
        row_count,
        seq_len,
        batch_size,
        embedding_dim,
        qkv_dim,
        head_count,
        head_dim,
        scale: 1.0 / (head_dim as f32).sqrt(),
        chunk_size: KDA_CHUNK_SIZE,
        decay_scale: KDA_DECAY_SCALE,
    }
}

pub(super) fn linear_config(element_count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (element_count.div_ceil(TC_BACKWARD_THREADS_PER_BLOCK), 1, 1),
        block_dim: (TC_BACKWARD_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(super) fn attention_config(seq_len: u32, head_count: u32, batch_size: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (seq_len, head_count, batch_size),
        block_dim: (SOFTMAX_D_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
