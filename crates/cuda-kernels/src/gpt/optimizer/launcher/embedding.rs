use cuda_core::DriverError;

use super::super::args::EmbeddingLookupGradArgs;
use super::super::threads::EMBEDDING_GRAD_THREADS_PER_BLOCK;
use super::OptimizerModule;
use crate::launch::linear_config;

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
            linear_config(len, EMBEDDING_GRAD_THREADS_PER_BLOCK),
            args.tokens,
            args.d_embedding_residual,
            args.d_token_embedding,
            args.token_count,
            args.embedding_dim,
        )
    }
}
