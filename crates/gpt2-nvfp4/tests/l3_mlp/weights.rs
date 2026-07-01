use cuda_core::CudaStream;
use gpt2_nvfp4::{
    HiddenVectorShape, MlpDownTensors, MlpDownWeightShape, MlpUpTensors, MlpUpWeightShape,
    MlpVectorShape,
};

use crate::common::upload::{upload_nvfp4_bytes, upload_zero_nvfp4, TestResult, UploadedNvfp4};
use crate::data::{mlp_down_identity_weight_bytes, mlp_up_repeat_weight_bytes};

pub struct WeightBuffers {
    up_weight: UploadedNvfp4,
    up_bias: UploadedNvfp4,
    down_weight: UploadedNvfp4,
    down_bias: UploadedNvfp4,
}

impl WeightBuffers {
    pub fn new(stream: &CudaStream) -> TestResult<Self> {
        Ok(Self {
            up_weight: upload_nvfp4_bytes::<MlpUpWeightShape>(stream, mlp_up_repeat_weight_bytes())?,
            up_bias: upload_zero_nvfp4::<MlpVectorShape>(stream)?,
            down_weight: upload_nvfp4_bytes::<MlpDownWeightShape>(
                stream,
                mlp_down_identity_weight_bytes(),
            )?,
            down_bias: upload_zero_nvfp4::<HiddenVectorShape>(stream)?,
        })
    }

    pub fn up_tensors(&self) -> MlpUpTensors<'_> {
        MlpUpTensors {
            weight: self.up_weight.mma(),
            bias: self.up_bias.device(),
        }
    }

    pub fn down_tensors(&self) -> MlpDownTensors<'_> {
        MlpDownTensors {
            weight: self.down_weight.mma(),
            bias: self.down_bias.device(),
        }
    }
}
