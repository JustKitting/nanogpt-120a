use super::args::{
    NextLatConcatArgs, NextLatConcatBackwardArgs, NextLatShape, NextLatSmoothL1Args,
};
use super::launcher::{NEXTLAT_THREADS_PER_BLOCK, NextLatModule};
use cuda_core::{DriverError, LaunchConfig};

impl NextLatModule {
    pub fn concat_input(&self, args: NextLatConcatArgs<'_, '_>) -> Result<(), DriverError> {
        self.core.nextlat_concat_input_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.row_count, 1, 1),
                block_dim: (NEXTLAT_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.next_token_embeddings,
            args.current_states,
            args.out,
            NextLatShape {
                row_count: args.row_count,
                embedding_dim: args.embedding_dim,
                seq_len: 0,
                batch_size: 0,
                lambda: 0.0,
            },
        )
    }

    pub fn concat_backward(
        &self,
        args: NextLatConcatBackwardArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.core.nextlat_concat_backward_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.row_count, 1, 1),
                block_dim: (NEXTLAT_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.d_concat,
            args.d_predicted,
            args.d_next_token_embeddings,
            args.d_current_states,
            NextLatShape {
                row_count: args.row_count,
                embedding_dim: args.embedding_dim,
                seq_len: 0,
                batch_size: 0,
                lambda: 0.0,
            },
        )
    }

    pub fn smooth_l1(&self, args: NextLatSmoothL1Args<'_, '_>) -> Result<(), DriverError> {
        self.core.nextlat_smooth_l1_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.batch_size, args.seq_len, 1),
                block_dim: (NEXTLAT_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.predicted_next_states,
            args.target_states,
            args.losses,
            args.d_predicted_next_states,
            NextLatShape {
                row_count: args.batch_size * args.seq_len,
                embedding_dim: args.embedding_dim,
                seq_len: args.seq_len,
                batch_size: args.batch_size,
                lambda: args.lambda,
            },
        )
    }
}
