use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::attention::AttentionModule;
use crate::launch::launch_config;

use super::{CAUSAL_ATTENTION_MAX_THREADS_PER_BLOCK, CausalAttentionParams};

pub struct CausalAttentionArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub qkv: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub log_sum_exp: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
}

impl AttentionModule {
    pub fn causal_attention(&self, args: CausalAttentionArgs<'_, '_>) -> Result<(), DriverError> {
        self.causal_attention.causal_attention_kernel(
            args.stream,
            launch_config(
                (args.seq_len, args.head_count, args.batch_size),
                causal_attention_threads(args.head_dim),
            ),
            args.qkv,
            args.out,
            args.log_sum_exp,
            CausalAttentionParams {
                row_count: args.row_count,
                seq_len: args.seq_len,
                batch_size: args.batch_size,
                embedding_dim: args.embedding_dim,
                qkv_dim: args.qkv_dim,
                head_count: args.head_count,
                head_dim: args.head_dim,
                scale: 1.0 / (args.head_dim as f32).sqrt(),
                chunk_size: 64,
                decay_scale: 0.01,
            },
        )
    }
}

fn causal_attention_threads(head_dim: u32) -> u32 {
    let threads = head_dim.div_ceil(32) * 32;
    assert!(threads <= CAUSAL_ATTENTION_MAX_THREADS_PER_BLOCK);
    threads.max(32)
}
