use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

#[path = "kernels/bias.rs"]
mod bias;
#[path = "kernels/projection.rs"]
mod projection;

pub(super) struct LoadedModule {
    pub(super) bias: bias::LoadedModule,
    pub(super) projection: projection::LoadedModule,
}

pub(super) fn from_module(module: Arc<CudaModule>) -> Result<LoadedModule, DriverError> {
    Ok(LoadedModule {
        bias: bias::from_module(module.clone())?,
        projection: projection::from_module(module)?,
    })
}
