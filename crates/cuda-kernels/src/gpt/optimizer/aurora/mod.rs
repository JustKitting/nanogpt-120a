//! Aurora device kernels.
//!
//! Algorithm ownership:
//! - `momentum`: builds the momentum buffer `M` and the update matrix used by Aurora.
//! - `polar`: computes `X0 = M / ||M||F` and polar-normalization setup.
//! - `row_balance`: computes and applies the diagonal row-balancing state `D_k`.
//! - `elementwise`: f32 glue kernels such as `out = alpha * a + beta * b`.
//! - `update`: applies the final Aurora update to the FP32 master weights.
//! - `reduce`: local block reductions shared by the Aurora kernels.

#[macro_use]
mod reduce;
pub(super) mod elementwise;
mod momentum;
pub(super) mod polar;
pub(super) mod row_balance;
pub(super) mod update;

use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

pub(super) struct LoadedModule {
    pub(super) momentum: momentum::module::LoadedModule,
    pub(super) elementwise: elementwise::module::LoadedModule,
    pub(super) polar: polar::module::LoadedModule,
    pub(super) row_balance: row_balance::module::LoadedModule,
    pub(super) update: update::module::LoadedModule,
}

pub(super) fn from_module(module: Arc<CudaModule>) -> Result<LoadedModule, DriverError> {
    Ok(LoadedModule {
        momentum: momentum::module::from_module(module.clone())?,
        elementwise: elementwise::module::from_module(module.clone())?,
        polar: polar::module::from_module(module.clone())?,
        row_balance: row_balance::module::from_module(module.clone())?,
        update: update::module::from_module(module)?,
    })
}
