use cuda_core::CudaStream;
use gpt2_nvfp4::{
    AttentionProjectionTensors, Gpt2BlockWeights, MlpProjectionTensors, Nvfp4Shape,
};

use super::{
    attention_projection_tensors, mlp_projection_tensors, upload_layer_norm, upload_nvfp4,
    TestResult, UploadedLayerNorm, UploadedLinear, UploadedPair,
};

pub struct UploadedBlock {
    pub ln_1: UploadedLayerNorm,
    pub attn_qkv: UploadedLinear,
    pub attn_c_proj: UploadedLinear,
    pub ln_2: UploadedLayerNorm,
    pub mlp_up: UploadedLinear,
    pub mlp_down: UploadedLinear,
}

impl UploadedBlock {
    pub fn attention_tensors(&self) -> AttentionProjectionTensors<'_> {
        attention_projection_tensors(&self.attn_qkv.weight, &self.attn_qkv.bias, &self.attn_c_proj.weight, &self.attn_c_proj.bias)
    }

    pub fn mlp_tensors(&self) -> MlpProjectionTensors<'_> {
        mlp_projection_tensors(&self.mlp_up.weight, &self.mlp_up.bias, &self.mlp_down.weight, &self.mlp_down.bias)
    }
}

pub fn upload_block(stream: &CudaStream, block: &Gpt2BlockWeights) -> TestResult<UploadedBlock> {
    Ok(UploadedBlock {
        ln_1: upload_layer_norm(stream, &block.ln_1)?,
        attn_qkv: upload_linear(stream, &block.attn.c_attn)?,
        attn_c_proj: upload_linear(stream, &block.attn.c_proj)?,
        ln_2: upload_layer_norm(stream, &block.ln_2)?,
        mlp_up: upload_linear(stream, &block.mlp.c_fc)?,
        mlp_down: upload_linear(stream, &block.mlp.c_proj)?,
    })
}

fn upload_linear<W: Nvfp4Shape, B: Nvfp4Shape>(stream: &CudaStream, linear: &gpt2_nvfp4::LinearWeights<W, B>) -> TestResult<UploadedLinear> {
    Ok(UploadedPair { weight: upload_nvfp4(stream, &linear.weight)?, bias: upload_nvfp4(stream, &linear.bias)? })
}
