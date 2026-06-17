mod causal;
mod causal_backward;
mod qkv_projection;

use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

pub use causal::{CausalAttentionArgs, CausalAttentionParams};
pub use causal_backward::{CausalAttentionBackwardArgs, CausalAttentionBackwardParams};
pub use qkv_projection::{CProjArgs, CProjTapeArgs, QkvProjectionArgs, QkvProjectionParams};

pub struct AttentionModule {
    qkv_projection: qkv_projection::kernels::LoadedModule,
    causal_attention: causal::kernels::LoadedModule,
    causal_attention_backward: causal_backward::kernels::LoadedModule,
}

impl AttentionModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            qkv_projection: qkv_projection::kernels::from_module(module.clone())?,
            causal_attention: causal::kernels::from_module(module.clone())?,
            causal_attention_backward: causal_backward::kernels::from_module(module)?,
        })
    }
}
