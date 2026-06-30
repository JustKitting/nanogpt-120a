use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

use super::kernels;

pub struct Nvfp4QuantModule {
    pub(super) row_amax: kernels::row_amax::module::LoadedModule,
    pub(super) four_six: kernels::four_six::module::LoadedModule,
    pub(super) ms_eden: kernels::ms_eden::module::LoadedModule,
    pub(super) ms_eden_amax: kernels::ms_eden::amax::module::LoadedModule,
    pub(super) ms_eden_fp32: kernels::ms_eden::fp32::module::LoadedModule,
    pub(super) ms_eden_fp32_pair: kernels::ms_eden::fp32_pair::module::LoadedModule,
    pub(super) ms_eden_fp32_transpose: kernels::ms_eden::fp32_transpose::module::LoadedModule,
    pub(super) ms_eden_nvfp4_transpose: kernels::ms_eden::nvfp4_transpose::module::LoadedModule,
}

impl Nvfp4QuantModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            row_amax: kernels::row_amax::module::from_module(module.clone())?,
            four_six: kernels::four_six::module::from_module(module.clone())?,
            ms_eden: kernels::ms_eden::module::from_module(module.clone())?,
            ms_eden_amax: kernels::ms_eden::amax::module::from_module(module.clone())?,
            ms_eden_fp32: kernels::ms_eden::fp32::module::from_module(module.clone())?,
            ms_eden_fp32_pair: kernels::ms_eden::fp32_pair::module::from_module(module.clone())?,
            ms_eden_fp32_transpose: kernels::ms_eden::fp32_transpose::module::from_module(
                module.clone(),
            )?,
            ms_eden_nvfp4_transpose: kernels::ms_eden::nvfp4_transpose::module::from_module(
                module,
            )?,
        })
    }
}
