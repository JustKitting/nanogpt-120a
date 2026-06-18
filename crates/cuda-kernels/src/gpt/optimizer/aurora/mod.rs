#[macro_use]
mod reduce;
pub(super) mod matrix;
mod momentum;
pub(super) mod row;
pub(super) mod update;

use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

pub(super) struct LoadedModule {
    pub(super) momentum: momentum::module::LoadedModule,
    pub(super) matrix: matrix::module::LoadedModule,
    pub(super) row: row::module::LoadedModule,
    pub(super) update: update::module::LoadedModule,
}

pub(super) fn from_module(module: Arc<CudaModule>) -> Result<LoadedModule, DriverError> {
    Ok(LoadedModule {
        momentum: momentum::module::from_module(module.clone())?,
        matrix: matrix::module::from_module(module.clone())?,
        row: row::module::from_module(module.clone())?,
        update: update::module::from_module(module)?,
    })
}
