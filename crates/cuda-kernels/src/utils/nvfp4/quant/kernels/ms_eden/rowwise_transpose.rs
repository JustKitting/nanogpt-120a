use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

#[path = "rowwise_transpose/no_pad.rs"]
pub(crate) mod no_pad;
#[path = "rowwise_transpose/padded.rs"]
pub(crate) mod padded;

pub(crate) struct LoadedModule {
    pub(crate) no_pad: no_pad::LoadedModule,
    pub(crate) padded: padded::LoadedModule,
}

pub(crate) fn from_module(module: Arc<CudaModule>) -> Result<LoadedModule, DriverError> {
    Ok(LoadedModule {
        no_pad: no_pad::from_module(module.clone())?,
        padded: padded::from_module(module)?,
    })
}
