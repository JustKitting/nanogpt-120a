use cuda_core::CudaStream;
use gpt2_nvfp4::{
    AttentionProjectionTensors, HiddenVectorShape, LayerNormTensors, Nvfp4Shape, QkvVectorShape,
    QkvWeightShape, ResidualWeightShape,
};

use crate::common::nvfp4::{E2M1_MIN_PAIR, E2M1_ONE_PAIR};
use crate::common::upload::{attention_projection_tensors, upload_nvfp4_bytes, upload_zero_nvfp4, TestResult, UploadedNvfp4};

pub struct WeightBuffers {
    qkv_weight: UploadedNvfp4,
    qkv_bias: UploadedNvfp4,
    c_proj_weight: UploadedNvfp4,
    c_proj_bias: UploadedNvfp4,
    ln_weight: UploadedNvfp4,
    ln_bias: UploadedNvfp4,
}

impl WeightBuffers {
    pub fn new(stream: &CudaStream) -> TestResult<Self> {
        Ok(Self {
            qkv_weight: upload_filled::<QkvWeightShape>(stream, E2M1_MIN_PAIR)?,
            qkv_bias: upload_zero_nvfp4::<QkvVectorShape>(stream)?,
            c_proj_weight: upload_filled::<ResidualWeightShape>(stream, E2M1_MIN_PAIR)?,
            c_proj_bias: upload_zero_nvfp4::<HiddenVectorShape>(stream)?,
            ln_weight: upload_filled::<HiddenVectorShape>(stream, E2M1_ONE_PAIR)?,
            ln_bias: upload_zero_nvfp4::<HiddenVectorShape>(stream)?,
        })
    }

    pub fn ln_1(&self) -> LayerNormTensors<'_> {
        LayerNormTensors {
            weight: self.ln_weight.device(),
            bias: self.ln_bias.device(),
        }
    }

    pub fn projections(&self) -> AttentionProjectionTensors<'_> {
        attention_projection_tensors(&self.qkv_weight, &self.qkv_bias, &self.c_proj_weight, &self.c_proj_bias)
    }
}

fn upload_filled<S: Nvfp4Shape>(stream: &CudaStream, byte: u8) -> TestResult<UploadedNvfp4> {
    upload_nvfp4_bytes::<S>(stream, vec![byte; S::BYTE_LEN])
}
