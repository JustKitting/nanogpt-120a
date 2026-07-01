use cuda_core::CudaStream;
use gpt2_nvfp4::NextLatWeights;

use crate::AppResult;

use super::{tensor::upload_nvfp4, UploadedLayerNorm, UploadedLinear};

pub struct UploadedNextLat {
    pub norm: UploadedLayerNorm,
    pub input_projection: UploadedLinear,
    pub transition: UploadedLinear,
    pub output_projection: UploadedLinear,
}

impl UploadedNextLat {
    pub(in crate::upload) fn new(stream: &CudaStream, weights: &NextLatWeights) -> AppResult<Self> {
        Ok(Self {
            norm: UploadedLayerNorm::from_parts(
                upload_nvfp4(stream, &weights.norm_weight)?,
                upload_nvfp4(stream, &weights.norm_bias)?,
            ),
            input_projection: UploadedLinear::from_linear(stream, &weights.input_projection)?,
            transition: UploadedLinear::from_linear(stream, &weights.transition)?,
            output_projection: UploadedLinear::from_linear(stream, &weights.output_projection)?,
        })
    }
}
