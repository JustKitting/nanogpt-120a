use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use super::{ApplyRopeParams, THREADS_PER_BLOCK};
use crate::attention::AttentionModule;

pub struct ApplyRopeArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub qkv: &'out mut DeviceBuffer<f32>,
    pub qkv_f16: Option<&'out mut DeviceBuffer<u16>>,
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
}

impl AttentionModule {
    pub fn apply_rope(&self, args: ApplyRopeArgs<'_, '_>) -> Result<(), DriverError> {
        let pair_count = args.batch_size * args.seq_len * args.head_count * (args.head_dim / 2);
        let config = LaunchConfig {
            grid_dim: (pair_count.div_ceil(THREADS_PER_BLOCK), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let params = ApplyRopeParams {
            row_count: args.row_count,
            seq_len: args.seq_len,
            batch_size: args.batch_size,
            embedding_dim: args.embedding_dim,
            qkv_dim: args.qkv_dim,
            head_count: args.head_count,
            head_dim: args.head_dim,
        };

        if let Some(qkv_f16) = args.qkv_f16 {
            return self.rope.apply_rope_save_f16_kernel(
                args.stream,
                config,
                args.qkv,
                qkv_f16,
                params,
            );
        }

        self.rope
            .apply_rope_kernel(args.stream, config, args.qkv, params)
    }
}
