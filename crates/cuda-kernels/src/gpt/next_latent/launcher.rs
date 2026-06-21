use super::{activation_kernels, kernels, projection_kernels};
use cuda_core::{CudaModule, DriverError};
use std::sync::Arc;
pub(super) const NEXTLAT_THREADS_PER_BLOCK: u32 = 256;

pub struct NextLatModule {
    pub(super) core: kernels::module::LoadedModule,
    pub(super) projection: projection_kernels::module::LoadedModule,
    pub(super) activation: activation_kernels::module::LoadedModule,
}

impl NextLatModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            core: kernels::module::from_module(module.clone())?,
            projection: projection_kernels::module::from_module(module.clone())?,
            activation: activation_kernels::module::from_module(module)?,
        })
    }
}
