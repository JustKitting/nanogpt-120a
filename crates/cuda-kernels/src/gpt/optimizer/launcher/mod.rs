mod adam;
mod aurora_mega;
mod embedding;
mod grad_clip;
mod kda_clip;
mod update;

use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError};

use super::modules;
use crate::nvfp4_quant::{Nvfp4QuantArgs, Nvfp4QuantModule, TensorAmaxArgs};

pub struct OptimizerModule {
    pub(super) apply: modules::LoadedModule,
    quant: Nvfp4QuantModule,
}

impl OptimizerModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            apply: modules::from_module(module.clone())?,
            quant: Nvfp4QuantModule::from_module(module)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn requantize(
        &self,
        stream: &CudaStream,
        bytes: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        global_scale: &mut DeviceBuffer<f32>,
        master: &DeviceBuffer<f32>,
        amax: &mut DeviceBuffer<f32>,
        chunk_amax: &mut DeviceBuffer<f32>,
        len: u32,
    ) -> Result<(), DriverError> {
        self.quant.tensor_amax_f32(TensorAmaxArgs {
            stream,
            x: master,
            chunk_amax,
            out: amax,
            element_count: len,
        })?;

        self.quant.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
            stream,
            x: master,
            amax: &*amax,
            out_fp4: bytes,
            out_scales: scales,
            out_global_scale: global_scale,
            group_count: len / 16,
        })
    }
}
