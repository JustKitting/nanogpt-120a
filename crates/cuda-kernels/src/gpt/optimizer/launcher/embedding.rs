use cuda_core::{DriverError, LaunchConfig};

use super::super::args::EmbeddingLookupGradArgs;
use super::super::threads::EMBEDDING_GRAD_THREADS_PER_BLOCK;
use super::OptimizerModule;

impl OptimizerModule {
    pub fn add_embedding_lookup_grad(
        &self,
        args: EmbeddingLookupGradArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let len = args.token_count * args.embedding_dim;
        assert!(args.tokens.len() >= args.token_count as usize);
        assert!(args.d_embedding_residual.len() >= len as usize);

        self.apply.embedding.embedding_lookup_grad_add_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (len.div_ceil(EMBEDDING_GRAD_THREADS_PER_BLOCK), 1, 1),
                block_dim: (EMBEDDING_GRAD_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.tokens,
            args.d_embedding_residual,
            args.d_token_embedding,
            args.token_count,
            args.embedding_dim,
        )
    }
}
