use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionProjectionTensors, HiddenVectorShape, LayerNormTensors, Nvfp4Shape, QkvVectorShape,
    QkvWeightShape, ResidualWeightShape,
};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

use crate::data::{E2M1_MIN_PAIR, E2M1_ONE_PAIR, E4M3_ONE};

pub struct WeightBuffers {
    qkv_weight_bytes: DeviceBuffer<u8>,
    qkv_weight_scales: DeviceBuffer<u8>,
    qkv_bias_bytes: DeviceBuffer<u8>,
    qkv_bias_scales: DeviceBuffer<u8>,
    c_proj_weight_bytes: DeviceBuffer<u8>,
    c_proj_weight_scales: DeviceBuffer<u8>,
    c_proj_bias_bytes: DeviceBuffer<u8>,
    c_proj_bias_scales: DeviceBuffer<u8>,
    ln_weight_bytes: DeviceBuffer<u8>,
    ln_weight_scales: DeviceBuffer<u8>,
    ln_bias_bytes: DeviceBuffer<u8>,
    ln_bias_scales: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
}

impl WeightBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            qkv_weight_bytes: filled_u8(stream, QkvWeightShape::BYTE_LEN, E2M1_MIN_PAIR)?,
            qkv_weight_scales: filled_u8(stream, QkvWeightShape::SCALE_LEN, E4M3_ONE)?,
            qkv_bias_bytes: filled_u8(stream, QkvVectorShape::BYTE_LEN, 0)?,
            qkv_bias_scales: filled_u8(stream, QkvVectorShape::SCALE_LEN, E4M3_ONE)?,
            c_proj_weight_bytes: filled_u8(stream, ResidualWeightShape::BYTE_LEN, E2M1_MIN_PAIR)?,
            c_proj_weight_scales: filled_u8(stream, ResidualWeightShape::SCALE_LEN, E4M3_ONE)?,
            c_proj_bias_bytes: filled_u8(stream, HiddenVectorShape::BYTE_LEN, 0)?,
            c_proj_bias_scales: filled_u8(stream, HiddenVectorShape::SCALE_LEN, E4M3_ONE)?,
            ln_weight_bytes: filled_u8(stream, HiddenVectorShape::BYTE_LEN, E2M1_ONE_PAIR)?,
            ln_weight_scales: filled_u8(stream, HiddenVectorShape::SCALE_LEN, E4M3_ONE)?,
            ln_bias_bytes: filled_u8(stream, HiddenVectorShape::BYTE_LEN, 0)?,
            ln_bias_scales: filled_u8(stream, HiddenVectorShape::SCALE_LEN, E4M3_ONE)?,
            global_scale: DeviceBuffer::from_host(stream, &[1.0_f32])?,
        })
    }

    pub fn ln_1(&self) -> LayerNormTensors<'_> {
        LayerNormTensors {
            weight: Nvfp4DeviceTensor::new(
                &self.ln_weight_bytes,
                &self.ln_weight_scales,
                &self.global_scale,
            ),
            bias: Nvfp4DeviceTensor::new(
                &self.ln_bias_bytes,
                &self.ln_bias_scales,
                &self.global_scale,
            ),
        }
    }

    pub fn projections(&self) -> AttentionProjectionTensors<'_> {
        AttentionProjectionTensors {
            qkv_weight: Nvfp4FourSixMmaWeightTensor::new(
                &self.qkv_weight_bytes,
                &self.qkv_weight_scales,
                &self.global_scale,
            ),
            qkv_bias: Nvfp4DeviceTensor::new(
                &self.qkv_bias_bytes,
                &self.qkv_bias_scales,
                &self.global_scale,
            ),
            c_proj_weight: Nvfp4FourSixMmaWeightTensor::new(
                &self.c_proj_weight_bytes,
                &self.c_proj_weight_scales,
                &self.global_scale,
            ),
            c_proj_bias: Nvfp4DeviceTensor::new(
                &self.c_proj_bias_bytes,
                &self.c_proj_bias_scales,
                &self.global_scale,
            ),
        }
    }
}

fn filled_u8(stream: &CudaStream, len: usize, value: u8) -> Result<DeviceBuffer<u8>, DriverError> {
    DeviceBuffer::from_host(stream, &vec![value; len])
}
