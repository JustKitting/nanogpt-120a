use cuda_core::{DriverError, LaunchConfig};

use super::super::args::{
    Nvfp4WeightUpdateArgs, ScheduleFreeAverageArgs, ScheduleFreeMaterializeArgs,
};
use super::super::threads::APPLY_THREADS_PER_BLOCK;
use super::OptimizerModule;

impl OptimizerModule {
    pub fn apply_nvfp4_weight_update(
        &self,
        args: Nvfp4WeightUpdateArgs<'_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.len % 16, 0);
        assert!(args.z_master.len() >= args.len as usize);
        assert!(args.x_master.len() >= args.len as usize);
        assert!(args.aurora_update.len() >= args.len as usize);

        self.apply.aurora.update.fp32_weight_update_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.len.div_ceil(APPLY_THREADS_PER_BLOCK), 1, 1),
                block_dim: (APPLY_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.z_master,
            args.aurora_update,
            args.learning_rate,
            args.weight_decay,
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

    pub fn materialize_schedule_free(
        &self,
        args: ScheduleFreeMaterializeArgs<'_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.len % 16, 0);
        assert!(args.z_master.len() >= args.len as usize);
        assert!(args.x_master.len() >= args.len as usize);
        assert!(args.materialized.len() >= args.len as usize);

        self.apply.schedule_free.schedule_free_interpolate_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.len.div_ceil(APPLY_THREADS_PER_BLOCK), 1, 1),
                block_dim: (APPLY_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.z_master,
            args.x_master,
            args.materialized,
            args.beta,
            args.len,
        )?;

        self.requantize(
            args.stream,
            args.bytes,
            args.scales,
            args.global_scale,
            &*args.materialized,
            args.amax,
            args.chunk_amax,
            args.len,
        )
    }

    pub fn update_schedule_free_average(
        &self,
        args: ScheduleFreeAverageArgs<'_>,
    ) -> Result<(), DriverError> {
        assert!(args.x_master.len() >= args.len as usize);
        assert!(args.z_master.len() >= args.len as usize);

        self.apply.schedule_free.schedule_free_average_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.len.div_ceil(APPLY_THREADS_PER_BLOCK), 1, 1),
                block_dim: (APPLY_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.x_master,
            args.z_master,
            args.coefficient,
            args.len,
        )
    }
}
