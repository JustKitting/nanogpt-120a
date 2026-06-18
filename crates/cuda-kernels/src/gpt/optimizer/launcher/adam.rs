use cuda_core::{DriverError, LaunchConfig};

use super::super::args::{AdamWUpdateArgs, ScheduleFreeAverageArgs};
use super::super::threads::APPLY_THREADS_PER_BLOCK;
use super::OptimizerModule;

impl OptimizerModule {
    pub fn apply_adamw_update(&self, args: AdamWUpdateArgs<'_>) -> Result<(), DriverError> {
        assert_eq!(args.len % 16, 0);
        assert!(args.z_master.len() >= args.len as usize);
        assert!(args.x_master.len() >= args.len as usize);
        assert!(args.grad.len() >= args.len as usize);
        assert!(args.first_moment.len() >= args.len as usize);
        assert!(args.second_moment.len() >= args.len as usize);

        self.apply.adam.fp32_adamw_update_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.len.div_ceil(APPLY_THREADS_PER_BLOCK), 1, 1),
                block_dim: (APPLY_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.z_master,
            args.grad,
            args.first_moment,
            args.second_moment,
            args.learning_rate,
            args.weight_decay,
            args.beta1,
            args.beta2,
            args.beta1_correction,
            args.beta2_correction,
            args.eps,
            args.len,
        )?;

        self.update_schedule_free_average(ScheduleFreeAverageArgs {
            stream: args.stream,
            x_master: args.x_master,
            z_master: &*args.z_master,
            len: args.len,
            coefficient: args.average_coefficient,
        })?;

        self.requantize(
            args.stream,
            args.bytes,
            args.scales,
            args.global_scale,
            &*args.x_master,
            args.amax,
            args.chunk_amax,
            args.len,
        )
    }
}
