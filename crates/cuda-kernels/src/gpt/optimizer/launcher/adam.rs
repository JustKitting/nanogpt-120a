use cuda_core::{DriverError, LaunchConfig};

use super::super::args::{AdamWUpdateArgs, Nvfp4WeightUpdateArgs};
use super::super::kernels::APPLY_THREADS_PER_BLOCK;
use super::OptimizerModule;

impl OptimizerModule {
    pub fn apply_adamw_update(&self, args: AdamWUpdateArgs<'_>) -> Result<(), DriverError> {
        assert_eq!(args.len % 16, 0);
        assert!(args.fp32_workspace.len() >= args.len as usize);
        assert!(args.grad.len() >= args.len as usize);
        assert!(args.first_moment.len() >= args.len as usize);
        assert!(args.second_moment.len() >= args.len as usize);

        self.apply.adam.nvfp4_adamw_update_to_f32_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.len.div_ceil(APPLY_THREADS_PER_BLOCK), 1, 1),
                block_dim: (APPLY_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            &*args.bytes,
            &*args.scales,
            args.grad,
            args.first_moment,
            args.second_moment,
            args.fp32_workspace,
            args.global_scale,
            args.learning_rate,
            args.weight_decay,
            args.beta1,
            args.beta2,
            args.beta1_correction,
            args.beta2_correction,
            args.eps,
            args.len,
        )?;

        self.requantize(Nvfp4WeightUpdateArgs {
            stream: args.stream,
            bytes: args.bytes,
            scales: args.scales,
            global_scale: args.global_scale,
            aurora_update: args.grad,
            fp32_workspace: args.fp32_workspace,
            amax: args.amax,
            next_global_scale: args.next_global_scale,
            len: args.len,
            learning_rate: args.learning_rate,
            weight_decay: args.weight_decay,
        })
    }
}
