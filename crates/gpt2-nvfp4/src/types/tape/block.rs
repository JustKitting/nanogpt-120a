use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::device_copy::copy_device;
use super::types::BlockForwardTape;
use crate::types::BlockForwardSaved;

impl<'a> BlockForwardTape<'a> {
    pub(super) fn saved(&self) -> BlockForwardSaved<'_> {
        BlockForwardSaved {
            residual_in: &*self.residual_in,
            ln_1: self.ln_1.saved(),
            qkv_input_nvfp4: self.qkv_input_nvfp4.saved(),
            qkv: &*self.qkv,
            attention_out: &*self.attention_out,
            c_proj_input_nvfp4: self.c_proj_input_nvfp4.saved(),
            residual_after_attention: &*self.residual_after_attention,
            ln_2: self.ln_2.saved(),
            mlp_up_input_nvfp4: self.mlp_up_input_nvfp4.saved(),
            mlp_up: &*self.mlp_up,
            mlp_relu2: &*self.mlp_relu2,
            mlp_down_input_nvfp4: self.mlp_down_input_nvfp4.saved(),
            residual_out: &*self.residual_out,
        }
    }

    pub(super) fn reborrow(&mut self) -> BlockForwardTape<'_> {
        BlockForwardTape {
            residual_in: &mut *self.residual_in,
            ln_1: self.ln_1.reborrow(),
            qkv_input_nvfp4: self.qkv_input_nvfp4.reborrow(),
            qkv: &mut *self.qkv,
            attention_out: &mut *self.attention_out,
            c_proj_input_nvfp4: self.c_proj_input_nvfp4.reborrow(),
            residual_after_attention: &mut *self.residual_after_attention,
            ln_2: self.ln_2.reborrow(),
            mlp_up_input_nvfp4: self.mlp_up_input_nvfp4.reborrow(),
            mlp_up: &mut *self.mlp_up,
            mlp_relu2: &mut *self.mlp_relu2,
            mlp_down_input_nvfp4: self.mlp_down_input_nvfp4.reborrow(),
            residual_out: &mut *self.residual_out,
        }
    }

    pub(crate) fn save_residual_in(
        &mut self,
        stream: &CudaStream,
        residual: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, residual, self.residual_in)
    }

    pub(crate) fn save_qkv(
        &mut self,
        stream: &CudaStream,
        qkv: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, qkv, self.qkv)
    }

    pub(crate) fn save_attention_out(
        &mut self,
        stream: &CudaStream,
        out: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, out, self.attention_out)
    }

    pub(crate) fn save_residual_after_attention(
        &mut self,
        stream: &CudaStream,
        residual: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, residual, self.residual_after_attention)
    }

    pub(crate) fn save_mlp_relu2(
        &mut self,
        stream: &CudaStream,
        activation: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, activation, self.mlp_relu2)
    }

    pub(crate) fn save_mlp_up(
        &mut self,
        stream: &CudaStream,
        activation: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, activation, self.mlp_up)
    }

    pub(crate) fn save_residual_out(
        &mut self,
        stream: &CudaStream,
        residual: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, residual, self.residual_out)
    }
}
