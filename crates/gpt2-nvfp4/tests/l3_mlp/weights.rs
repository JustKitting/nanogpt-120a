use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    HiddenVectorShape, MlpDownTensors, MlpDownWeightShape, MlpUpTensors, MlpUpWeightShape,
    MlpVectorShape, Nvfp4Shape,
};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

use crate::data::{mlp_down_identity_weight_bytes, mlp_up_repeat_weight_bytes};
use crate::nvfp4_common::filled_u8;

const E4M3_ONE: u8 = 0x38;

pub struct WeightBuffers {
    up_weight_bytes: DeviceBuffer<u8>,
    up_weight_scales: DeviceBuffer<u8>,
    up_bias_bytes: DeviceBuffer<u8>,
    up_bias_scales: DeviceBuffer<u8>,
    down_weight_bytes: DeviceBuffer<u8>,
    down_weight_scales: DeviceBuffer<u8>,
    down_bias_bytes: DeviceBuffer<u8>,
    down_bias_scales: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
}

impl WeightBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            up_weight_bytes: DeviceBuffer::from_host(stream, &mlp_up_repeat_weight_bytes())?,
            up_weight_scales: filled_u8(stream, MlpUpWeightShape::SCALE_LEN, E4M3_ONE)?,
            up_bias_bytes: filled_u8(stream, MlpVectorShape::BYTE_LEN, 0)?,
            up_bias_scales: filled_u8(stream, MlpVectorShape::SCALE_LEN, E4M3_ONE)?,
            down_weight_bytes: DeviceBuffer::from_host(stream, &mlp_down_identity_weight_bytes())?,
            down_weight_scales: filled_u8(stream, MlpDownWeightShape::SCALE_LEN, E4M3_ONE)?,
            down_bias_bytes: filled_u8(stream, HiddenVectorShape::BYTE_LEN, 0)?,
            down_bias_scales: filled_u8(stream, HiddenVectorShape::SCALE_LEN, E4M3_ONE)?,
            global_scale: DeviceBuffer::from_host(stream, &[1.0_f32])?,
        })
    }

    pub fn up_tensors(&self) -> MlpUpTensors<'_> {
        MlpUpTensors {
            weight: Nvfp4FourSixMmaWeightTensor::new(
                &self.up_weight_bytes,
                &self.up_weight_scales,
                &self.global_scale,
            ),
            bias: Nvfp4DeviceTensor::new(
                &self.up_bias_bytes,
                &self.up_bias_scales,
                &self.global_scale,
            ),
        }
    }

    pub fn down_tensors(&self) -> MlpDownTensors<'_> {
        MlpDownTensors {
            weight: Nvfp4FourSixMmaWeightTensor::new(
                &self.down_weight_bytes,
                &self.down_weight_scales,
                &self.global_scale,
            ),
            bias: Nvfp4DeviceTensor::new(
                &self.down_bias_bytes,
                &self.down_bias_scales,
                &self.global_scale,
            ),
        }
    }
}
