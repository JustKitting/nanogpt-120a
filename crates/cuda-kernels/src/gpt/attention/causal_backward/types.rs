use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};

use crate::attention::AttentionModule;

pub const CAUSAL_BACKWARD_HEAD_DIM_THREADS: u32 = 64;
pub const CAUSAL_BACKWARD_KEY_BLOCK: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CausalAttentionBackwardParams {
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
    pub scale: f32,
}

unsafe impl DeviceCopy for CausalAttentionBackwardParams {}

pub struct CausalAttentionBackwardArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub qkv: &'a DeviceBuffer<f32>,
    pub attention_out: &'a DeviceBuffer<f32>,
    pub d_out: &'a DeviceBuffer<f32>,
    pub log_sum_exp: &'a DeviceBuffer<f32>,
    pub softmax_d: &'scratch mut DeviceBuffer<f32>,
    pub d_qkv: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
}

impl AttentionModule {
    pub fn causal_attention_backward(
        &self,
        args: CausalAttentionBackwardArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        let params = params(&args);
        self.causal_attention_backward.softmax_d_kernel(
            args.stream,
            config(args.seq_len, args.head_count, args.batch_size),
            args.attention_out,
            args.d_out,
            args.softmax_d,
            params,
        )?;
        self.causal_attention_backward.dq_kernel(
            args.stream,
            config(args.seq_len, args.head_count, args.batch_size),
            args.qkv,
            args.d_out,
            args.log_sum_exp,
            args.softmax_d,
            args.d_qkv,
            params,
        )?;
        self.causal_attention_backward.dkv_kernel(
            args.stream,
            dkv_config(args.seq_len, args.head_count, args.batch_size),
            args.qkv,
            args.d_out,
            args.log_sum_exp,
            args.softmax_d,
            args.d_qkv,
            params,
        )
    }
}

fn params(args: &CausalAttentionBackwardArgs<'_, '_, '_>) -> CausalAttentionBackwardParams {
    CausalAttentionBackwardParams {
        row_count: args.row_count,
        seq_len: args.seq_len,
        batch_size: args.batch_size,
        embedding_dim: args.embedding_dim,
        qkv_dim: args.qkv_dim,
        head_count: args.head_count,
        head_dim: args.head_dim,
        scale: 1.0 / (args.head_dim as f32).sqrt(),
    }
}

fn config(seq_len: u32, head_count: u32, batch_size: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (seq_len, head_count, batch_size),
        block_dim: (CAUSAL_BACKWARD_HEAD_DIM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn dkv_config(seq_len: u32, head_count: u32, batch_size: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            seq_len.div_ceil(CAUSAL_BACKWARD_KEY_BLOCK),
            head_count,
            batch_size,
        ),
        block_dim: (
            CAUSAL_BACKWARD_KEY_BLOCK * CAUSAL_BACKWARD_HEAD_DIM_THREADS,
            1,
            1,
        ),
        shared_mem_bytes: 0,
    }
}
