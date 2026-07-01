use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

mod base;
pub(in crate::attention) mod kda;

pub(crate) struct LoadedModule {
    pub(super) base: base::LoadedModule,
    pub(in crate::attention) kda: kda::LoadedModule,
}

pub(crate) fn from_module(module: Arc<CudaModule>) -> Result<LoadedModule, DriverError> {
    Ok(LoadedModule {
        base: base::from_module(module.clone())?,
        kda: kda::from_module(module)?,
    })
}
