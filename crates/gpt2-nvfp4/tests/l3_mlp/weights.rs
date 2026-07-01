use cuda_core::CudaStream;
use gpt2_nvfp4::{
    HiddenVectorShape, MlpDownWeightShape, MlpProjectionTensors, MlpUpWeightShape, MlpVectorShape,
};

use crate::common::upload::{mlp_projection_tensors, upload_nvfp4_bytes, upload_zero_nvfp4, TestResult, UploadedNvfp4};
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

    pub fn projections(&self) -> MlpProjectionTensors<'_> {
        mlp_projection_tensors(&self.up_weight, &self.up_bias, &self.down_weight, &self.down_bias)
    }
}
