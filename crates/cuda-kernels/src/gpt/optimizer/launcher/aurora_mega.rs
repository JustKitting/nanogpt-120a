use cuda_core::{DriverError, LaunchConfig};

use crate::f16_tc_matmul::cta_tile::CTA_THREADS;

use super::super::args::AuroraMegaUpdateArgs;
use super::super::{AURORA_COOPERATIVE_BLOCKS, AURORA_MATRIX_PHASES};
use super::OptimizerModule;

impl OptimizerModule {
    pub fn aurora_mega_update(&self, args: AuroraMegaUpdateArgs<'_>) -> Result<(), DriverError> {
        assert_mega_args(&args);
        let matrix_count = args.slot_count / AURORA_MATRIX_PHASES as u32;
        self.apply
            .aurora
            .mega
            .aurora_mega_update_cooperative_kernel(
                args.stream,
                LaunchConfig {
                    grid_dim: (AURORA_COOPERATIVE_BLOCKS as u32, matrix_count, 1),
                    block_dim: (CTA_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                },
                args.grad_ptrs,
                args.momentum_ptrs,
                args.z_master_ptrs,
                args.x_master_ptrs,
                args.byte_ptrs,
                args.scale_ptrs,
                args.global_scale_ptrs,
                args.rows,
                args.cols,
                args.oriented,
                args.polar_next,
                args.polar_x,
                args.polar_gram,
                args.polar_ax,
                args.polar_chunks,
                args.slot_count,
                args.max_len,
                args.max_ax_len,
                args.max_dim,
                args.mu,
                args.learning_rate,
                args.weight_decay,
                args.average_coefficient,
                args.iterations,
            )
    }
}

fn assert_mega_args(args: &AuroraMegaUpdateArgs<'_>) {
    let slots = args.slot_count as usize;
    let matrix_count = args.slot_count as usize / AURORA_MATRIX_PHASES;
    assert_eq!(args.slot_count as usize % AURORA_MATRIX_PHASES, 0);
    assert!(args.grad_ptrs.len() >= slots);
    assert!(args.momentum_ptrs.len() >= slots);
    assert!(args.z_master_ptrs.len() >= slots);
    assert!(args.x_master_ptrs.len() >= slots);
    assert!(args.byte_ptrs.len() >= slots);
    assert!(args.scale_ptrs.len() >= slots);
    assert!(args.global_scale_ptrs.len() >= slots);
    assert!(args.rows.len() >= slots);
    assert!(args.cols.len() >= slots);
    assert!(args.oriented.len() >= args.max_len as usize * matrix_count);
    assert!(args.polar_next.len() >= args.max_len as usize * matrix_count);
    assert!(args.polar_x.len() >= args.max_len as usize * matrix_count);
    assert!(args.polar_ax.len() >= args.max_ax_len as usize * matrix_count);
    assert!(args.polar_gram.len() >= args.max_dim as usize * args.max_dim as usize * matrix_count);
    assert!(args.polar_chunks.len() >= AURORA_COOPERATIVE_BLOCKS * matrix_count);
}
