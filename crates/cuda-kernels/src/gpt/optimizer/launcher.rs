use std::sync::Arc;

use cuda_core::{CudaModule, DriverError, LaunchConfig};

use super::args::Nvfp4WeightUpdateArgs;
use super::kernels::{self, APPLY_THREADS_PER_BLOCK};
use crate::nvfp4_quant::{Nvfp4QuantArgs, Nvfp4QuantModule, RowAmaxArgs};

pub struct OptimizerModule {
    apply: kernels::LoadedModule,
    quant: Nvfp4QuantModule,
}

impl OptimizerModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            apply: kernels::from_module(module.clone())?,
            quant: Nvfp4QuantModule::from_module(module)?,
        })
    }

    pub fn apply_nvfp4_weight_update(
        &self,
        args: Nvfp4WeightUpdateArgs<'_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.len % 16, 0);
        assert!(args.fp32_workspace.len() >= args.len as usize);
        assert!(args.aurora_update.len() >= args.len as usize);

        self.apply.nvfp4_weight_update_to_f32_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.len.div_ceil(APPLY_THREADS_PER_BLOCK), 1, 1),
                block_dim: (APPLY_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            &*args.bytes,
            &*args.scales,
            args.aurora_update,
            args.fp32_workspace,
            args.global_scale,
            args.learning_rate,
            args.weight_decay,
            args.len,
        )?;

        self.quant.row_amax_f32(RowAmaxArgs {
            stream: args.stream,
            x: &*args.fp32_workspace,
            out: args.amax,
            row_count: 1,
            row_len: args.len,
        })?;

        self.quant.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
            stream: args.stream,
            x: &*args.fp32_workspace,
            amax: &*args.amax,
            out_fp4: args.bytes,
            out_scales: args.scales,
            out_global_scale: args.next_global_scale,
            group_count: args.len / 16,
        })
    }
}
