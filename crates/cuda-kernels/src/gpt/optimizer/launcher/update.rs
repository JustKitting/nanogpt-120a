use cuda_core::{DriverError, LaunchConfig};

use super::super::args::Nvfp4WeightUpdateArgs;
use super::super::kernels::APPLY_THREADS_PER_BLOCK;
use super::OptimizerModule;

impl OptimizerModule {
    pub fn apply_nvfp4_weight_update(
        &self,
        args: Nvfp4WeightUpdateArgs<'_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.len % 16, 0);
        assert!(args.master.len() >= args.len as usize);
        assert!(args.aurora_update.len() >= args.len as usize);

        self.apply.update.fp32_weight_update_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.len.div_ceil(APPLY_THREADS_PER_BLOCK), 1, 1),
                block_dim: (APPLY_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.master,
            args.aurora_update,
            args.learning_rate,
            args.weight_decay,
            args.len,
        )?;

        self.requantize(
            args.stream,
            args.bytes,
            args.scales,
            args.global_scale,
            &*args.master,
            args.amax,
            args.chunk_amax,
            args.len,
        )
    }
}
