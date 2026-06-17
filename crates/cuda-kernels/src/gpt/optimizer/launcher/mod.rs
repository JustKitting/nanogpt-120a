mod adam;
mod aurora;
mod aurora_scale;
mod embedding;
mod update;

use std::sync::Arc;

use cuda_core::{CudaModule, DriverError, LaunchConfig};

use super::kernels::{self, MATRIX_THREADS_PER_BLOCK};
use crate::nvfp4_quant::{Nvfp4QuantArgs, Nvfp4QuantModule, RowAmaxArgs};

pub struct OptimizerModule {
    pub(super) apply: kernels::LoadedModule,
    quant: Nvfp4QuantModule,
}

impl OptimizerModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            apply: kernels::from_module(module.clone())?,
            quant: Nvfp4QuantModule::from_module(module)?,
        })
    }

    fn requantize(&self, args: super::args::Nvfp4WeightUpdateArgs<'_>) -> Result<(), DriverError> {
        self.quant.row_amax_f32(RowAmaxArgs {
            stream: args.stream,
            x: &*args.fp32_workspace,
            out: args.amax,
            row_count: 1,
            row_len: args.len,
        })?;

        self.quant.fp32_to_nvfp4_four_six_fixed_global(
            Nvfp4QuantArgs {
                stream: args.stream,
                x: &*args.fp32_workspace,
                amax: &*args.amax,
                out_fp4: args.bytes,
                out_scales: args.scales,
                out_global_scale: args.next_global_scale,
                group_count: args.len / 16,
            },
            args.requantize_global_scale,
        )
    }
}

pub(super) fn matrix_config(len: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (len.div_ceil(MATRIX_THREADS_PER_BLOCK), 1, 1),
        block_dim: (MATRIX_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
