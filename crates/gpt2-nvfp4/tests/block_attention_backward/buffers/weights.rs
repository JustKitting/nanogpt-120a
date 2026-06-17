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
        })
    }

    pub fn ln_1(&self) -> LayerNormTensors<'_> {
        LayerNormTensors {
            weight: nvfp4_device(&self.ln_weight_bytes, &self.ln_weight_scales),
            bias: nvfp4_device(&self.ln_bias_bytes, &self.ln_bias_scales),
        }
    }

    pub fn projections(&self) -> AttentionProjectionTensors<'_> {
        AttentionProjectionTensors {
            qkv_weight: mma_weight(&self.qkv_weight_bytes, &self.qkv_weight_scales),
            qkv_bias: nvfp4_device(&self.qkv_bias_bytes, &self.qkv_bias_scales),
            c_proj_weight: mma_weight(&self.c_proj_weight_bytes, &self.c_proj_weight_scales),
            c_proj_bias: nvfp4_device(&self.c_proj_bias_bytes, &self.c_proj_bias_scales),
        }
    }
}

fn filled_u8(stream: &CudaStream, len: usize, value: u8) -> Result<DeviceBuffer<u8>, DriverError> {
    DeviceBuffer::from_host(stream, &vec![value; len])
}

fn mma_weight<'a>(
    bytes: &'a DeviceBuffer<u8>,
    scales: &'a DeviceBuffer<u8>,
) -> Nvfp4FourSixMmaWeightTensor<'a> {
    Nvfp4FourSixMmaWeightTensor {
        bytes,
        scales,
        global_scale: 1.0,
    }
}

fn nvfp4_device<'a>(
    bytes: &'a DeviceBuffer<u8>,
    scales: &'a DeviceBuffer<u8>,
) -> Nvfp4DeviceTensor<'a> {
    Nvfp4DeviceTensor {
        bytes,
        scales,
        global_scale: 1.0,
    }
}
