use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

use super::args::{
    ARGMAX_THREADS_PER_BLOCK, LOGITS_TOP_K, LogitsArgmaxArgs, LogitsArgmaxParams, LogitsTopKArgs,
    LogitsTopKParams, TOPK_THREADS_PER_BLOCK,
};
use super::kernels::kernels;
use crate::launch::grid_x_config;

pub struct LogitsModule {
    module: kernels::LoadedModule,
}

impl LogitsModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn argmax(&self, args: LogitsArgmaxArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.logits_argmax_kernel(
            args.stream,
            grid_x_config(1, ARGMAX_THREADS_PER_BLOCK),
            args.logits,
            args.out_token,
            LogitsArgmaxParams {
                row: args.row,
                vocab_size: args.vocab_size,
            },
        )
    }

    pub fn top_k(&self, args: LogitsTopKArgs<'_, '_>) -> Result<(), DriverError> {
        let k = args.k.clamp(1, LOGITS_TOP_K as u32);
        self.module.logits_top_k_kernel(
            args.stream,
            grid_x_config(1, TOPK_THREADS_PER_BLOCK),
            args.logits,
            args.out_tokens,
            args.out_values,
            LogitsTopKParams {
                row: args.row,
                vocab_size: args.vocab_size,
                k,
            },
        )
    }
}
