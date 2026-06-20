use cuda_core::{DriverError, LaunchConfig};

use super::super::args::GradientClipArgs;
use super::super::grad_clip::GRAD_CLIP_THREADS_PER_BLOCK;
use super::OptimizerModule;

impl OptimizerModule {
    pub fn clip_gradients(&self, args: GradientClipArgs<'_>) -> Result<(), DriverError> {
        assert!(args.ptrs.len() >= args.slot_count as usize);
        assert!(args.lens.len() >= args.slot_count as usize);
        assert!(args.chunk_offsets.len() >= args.slot_count as usize);
        assert!(args.chunk_sums.len() >= args.chunk_count as usize);
        assert!(!args.scale.is_empty());

        let chunk_grid = (args.chunk_count, 1, 1);
        let chunk_block = (GRAD_CLIP_THREADS_PER_BLOCK, 1, 1);
        self.apply.grad_clip.grad_clip_sumsq_chunks_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: chunk_grid,
                block_dim: chunk_block,
                shared_mem_bytes: 0,
            },
            args.ptrs,
            args.lens,
            args.chunk_offsets,
            args.chunk_sums,
            args.slot_count,
            args.chunk_count,
        )?;

        self.apply.grad_clip.grad_clip_scale_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: chunk_block,
                shared_mem_bytes: 0,
            },
            args.chunk_sums,
            args.scale,
            args.chunk_count,
            args.max_norm,
        )?;

        self.apply.grad_clip.grad_clip_apply_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: chunk_grid,
                block_dim: chunk_block,
                shared_mem_bytes: 0,
            },
            args.ptrs,
            args.lens,
            args.chunk_offsets,
            args.scale,
            args.slot_count,
            args.chunk_count,
        )
    }
}
