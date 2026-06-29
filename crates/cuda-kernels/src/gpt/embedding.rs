use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::layer_norm_utils::{
    layer_norm_columns3, layer_norm_map3, layer_norm_store3, nvfp4_column,
};
use crate::nvfp4::Nvfp4DeviceTensor;

const EMBEDDING_THREADS_PER_BLOCK: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmbeddingParams {
    pub hidden_len: u32,
    pub embedding_dim: u32,
}

unsafe impl DeviceCopy for EmbeddingParams {}

pub struct EmbeddingArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub tokens: &'a DeviceBuffer<u32>,
    pub token_embedding: Nvfp4DeviceTensor<'a>,
    pub residual: &'out mut DeviceBuffer<f32>,
    pub hidden_len: u32,
    pub embedding_dim: u32,
}

pub struct EmbeddingModule {
    module: kernels::LoadedModule,
}

impl EmbeddingModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn token_embedding(&self, args: EmbeddingArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.token_embedding_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.hidden_len / args.embedding_dim, 1, 1),
                block_dim: (EMBEDDING_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.tokens,
            args.token_embedding.bytes,
            args.token_embedding.scales,
            args.token_embedding.global_scale,
            args.residual,
            EmbeddingParams {
                hidden_len: args.hidden_len,
                embedding_dim: args.embedding_dim,
            },
        )
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn token_embedding_kernel(
        tokens: &[u32],
        token_embedding_bytes: &[u8],
        token_embedding_scales: &[u8],
        token_embedding_global_scale: &[f32],
        mut residual: DisjointSlice<f32>,
        params: EmbeddingParams,
    ) {
        let row = thread::blockIdx_x();
        let thread = thread::threadIdx_x();

        if row < params.hidden_len / params.embedding_dim {
            let token = tokens[row as usize];
            let row_base = row as usize * params.embedding_dim as usize;
            let token_base = token as usize * params.embedding_dim as usize;

            let cols = layer_norm_columns3!(thread, EMBEDDING_THREADS_PER_BLOCK);
            let values = layer_norm_map3!(cols, |col| nvfp4_column(
                token_embedding_bytes,
                token_embedding_scales,
                token_embedding_global_scale[0],
                token_base,
                col,
                params.embedding_dim,
            ));

            layer_norm_store3!(&mut residual, row_base, cols, params.embedding_dim, values);
        }
    }
}
