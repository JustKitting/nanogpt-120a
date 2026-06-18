use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

use super::{adam, aurora, embedding, schedule_free};

pub(super) struct LoadedModule {
    pub(super) adam: adam::module::LoadedModule,
    pub(super) aurora: aurora::LoadedModule,
    pub(super) embedding: embedding::module::LoadedModule,
    pub(super) schedule_free: schedule_free::module::LoadedModule,
}

pub(super) fn from_module(module: Arc<CudaModule>) -> Result<LoadedModule, DriverError> {
    Ok(LoadedModule {
        adam: adam::module::from_module(module.clone())?,
        aurora: aurora::from_module(module.clone())?,
        embedding: embedding::module::from_module(module.clone())?,
        schedule_free: schedule_free::module::from_module(module)?,
    })
}
