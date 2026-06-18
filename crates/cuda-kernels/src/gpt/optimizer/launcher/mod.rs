mod adam;
mod aurora;
mod aurora_momentum;
mod aurora_scale;
mod embedding;
mod update;

use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use super::modules;
use super::threads::MATRIX_THREADS_PER_BLOCK;
use crate::nvfp4_quant::{
    Nvfp4QuantArgs, Nvfp4QuantModule, TensorAmaxArgs, nvfp4_tensor_amax_chunks,
};

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

    #[allow(clippy::too_many_arguments)]
    fn requantize_from_chunk_amax(
        &self,
        stream: &CudaStream,
        bytes: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        global_scale: &mut DeviceBuffer<f32>,
        master: &DeviceBuffer<f32>,
        amax: &mut DeviceBuffer<f32>,
        chunk_amax: &DeviceBuffer<f32>,
        len: u32,
    ) -> Result<(), DriverError> {
        let chunk_count = nvfp4_tensor_amax_chunks(len as usize) as u32;
        self.quant
            .tensor_amax_from_chunks_f32(stream, chunk_amax, amax, chunk_count)?;

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

pub(super) fn matrix_config(len: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (len.div_ceil(MATRIX_THREADS_PER_BLOCK), 1, 1),
        block_dim: (MATRIX_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
