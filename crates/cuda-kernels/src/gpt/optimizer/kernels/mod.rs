use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

#[macro_use]
mod reduce;
pub(super) mod adam;
pub(super) mod aurora;
pub(super) mod embedding;
pub(super) mod matrix;
pub(super) mod row;
pub(super) mod update;

pub const APPLY_THREADS_PER_BLOCK: u32 = 256;
pub const EMBEDDING_GRAD_THREADS_PER_BLOCK: u32 = 256;
pub const MATRIX_THREADS_PER_BLOCK: u32 = 256;
pub(super) const WARP_SIZE: u32 = 32;
pub(super) const WARPS_PER_BLOCK: u32 = MATRIX_THREADS_PER_BLOCK / WARP_SIZE;

pub(super) struct LoadedModule {
    pub(super) adam: adam::module::LoadedModule,
    pub(super) aurora: aurora::module::LoadedModule,
    pub(super) embedding: embedding::module::LoadedModule,
    pub(super) matrix: matrix::module::LoadedModule,
    pub(super) row: row::module::LoadedModule,
    pub(super) update: update::module::LoadedModule,
}

pub(super) fn from_module(module: Arc<CudaModule>) -> Result<LoadedModule, DriverError> {
    Ok(LoadedModule {
        adam: adam::module::from_module(module.clone())?,
        aurora: aurora::module::from_module(module.clone())?,
        embedding: embedding::module::from_module(module.clone())?,
        matrix: matrix::module::from_module(module.clone())?,
        row: row::module::from_module(module.clone())?,
        update: update::module::from_module(module)?,
    })
}
