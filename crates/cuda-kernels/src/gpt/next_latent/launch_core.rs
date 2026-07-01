use super::args::{
    NextLatConcatArgs, NextLatConcatBackwardArgs, NextLatShape, NextLatSmoothL1Args,
};
use super::launcher::{NEXTLAT_THREADS_PER_BLOCK, NextLatModule};
use crate::launch::{grid_x_config, launch_config};
use cuda_core::DriverError;

impl NextLatModule {
    pub fn concat_input(&self, args: NextLatConcatArgs<'_, '_>) -> Result<(), DriverError> {
        self.core.nextlat_concat_input_kernel(
            args.stream,
            grid_x_config(args.row_count, NEXTLAT_THREADS_PER_BLOCK),
            args.next_token_embeddings,
            args.current_states,
            args.out,
            NextLatShape::rows(args.row_count, args.embedding_dim),
        )
    }

    pub fn concat_backward(
        &self,
        args: NextLatConcatBackwardArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.core.nextlat_concat_backward_kernel(
            args.stream,
            grid_x_config(args.row_count, NEXTLAT_THREADS_PER_BLOCK),
            args.d_concat,
            args.d_predicted,
            args.d_next_token_embeddings,
            args.d_current_states,
            NextLatShape::rows(args.row_count, args.embedding_dim),
        )
    }

    pub fn smooth_l1(&self, args: NextLatSmoothL1Args<'_, '_>) -> Result<(), DriverError> {
        self.core.nextlat_smooth_l1_kernel(
            args.stream,
            launch_config(
                (args.batch_size, args.seq_len, 1),
                NEXTLAT_THREADS_PER_BLOCK,
            ),
            args.predicted_next_states,
            args.target_states,
            args.losses,
            args.d_predicted_next_states,
            NextLatShape::smooth_l1(
                args.batch_size,
                args.seq_len,
                args.embedding_dim,
                args.lambda,
            ),
        )
    }
}
