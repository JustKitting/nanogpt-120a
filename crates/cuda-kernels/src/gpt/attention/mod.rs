mod causal;
mod causal_backward_tc;
mod qkv_projection;

use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

pub use causal::{CausalAttentionArgs, CausalAttentionParams};
pub use causal_backward_tc::{CausalAttentionBackwardTcArgs, CausalAttentionBackwardTcScratch};
pub use qkv_projection::{CProjArgs, CProjTapeArgs, QkvProjectionArgs, QkvProjectionParams};

pub struct AttentionModule {
    qkv_projection: qkv_projection::kernels::LoadedModule,
    causal_attention: causal::kernels::LoadedModule,
    causal_attention_backward_tc: causal_backward_tc::kernels::LoadedModule,
}

impl AttentionModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            qkv_projection: qkv_projection::kernels::from_module(module.clone())?,
            causal_attention: causal::kernels::from_module(module.clone())?,
            causal_attention_backward_tc: causal_backward_tc::kernels::from_module(module)?,
        })
    }
}
