use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::{
    Gpt2BlockWeights, Gpt2Weights, LayerNormTensors, LayerNormWeights, LinearWeights, Nvfp4Shape,
    Nvfp4Tensor,
};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

use crate::AppResult;

pub struct UploadedModel {
    pub token_embedding: UploadedNvfp4,
    pub blocks: Vec<UploadedBlock>,
    pub ln_f: UploadedLayerNorm,
}

impl UploadedModel {
    pub fn new(stream: &CudaStream, weights: &Gpt2Weights) -> AppResult<Self> {
        Ok(Self {
            token_embedding: upload_nvfp4(stream, &weights.embeddings.wte)?,
            blocks: weights
                .h
                .iter()
                .map(|block| UploadedBlock::new(stream, block))
                .collect::<AppResult<_>>()?,
            ln_f: UploadedLayerNorm::new(stream, &weights.ln_f)?,
        })
    }
}

pub struct UploadedBlock {
    pub ln_1: UploadedLayerNorm,
    pub attn_qkv: UploadedLinear,
    pub attn_c_proj: UploadedLinear,
    pub ln_2: UploadedLayerNorm,
    pub mlp_up: UploadedLinear,
    pub mlp_down: UploadedLinear,
}

impl UploadedBlock {
    fn new(stream: &CudaStream, block: &Gpt2BlockWeights) -> AppResult<Self> {
        Ok(Self {
            ln_1: UploadedLayerNorm::new(stream, &block.ln_1)?,
            attn_qkv: upload_linear(stream, &block.attn.c_attn)?,
            attn_c_proj: upload_linear(stream, &block.attn.c_proj)?,
            ln_2: UploadedLayerNorm::new(stream, &block.ln_2)?,
            mlp_up: upload_linear(stream, &block.mlp.c_fc)?,
            mlp_down: upload_linear(stream, &block.mlp.c_proj)?,
        })
    }
}

pub struct UploadedLinear {
    pub weight: UploadedNvfp4,
    pub bias: UploadedNvfp4,
}
pub struct UploadedLayerNorm {
    pub(crate) weight: UploadedNvfp4,
    pub(crate) bias: UploadedNvfp4,
}

impl UploadedLayerNorm {
    fn new(stream: &CudaStream, layer_norm: &LayerNormWeights) -> AppResult<Self> {
        Ok(Self {
            weight: upload_nvfp4(stream, &layer_norm.weight)?,
            bias: upload_nvfp4(stream, &layer_norm.bias)?,
        })
    }

    pub fn tensors(&self) -> LayerNormTensors<'_> {
        LayerNormTensors {
            weight: self.weight.device(),
            bias: self.bias.device(),
        }
    }
}
pub struct UploadedNvfp4 {
    pub(crate) bytes: DeviceBuffer<u8>,
    pub(crate) scales: DeviceBuffer<u8>,
    pub(crate) global_scale: f32,
}

impl UploadedNvfp4 {
    pub fn device(&self) -> Nvfp4DeviceTensor<'_> {
        Nvfp4DeviceTensor {
            bytes: &self.bytes,
            scales: &self.scales,
            global_scale: self.global_scale,
        }
    }

    pub fn mma(&self) -> Nvfp4FourSixMmaWeightTensor<'_> {
        Nvfp4FourSixMmaWeightTensor {
            bytes: &self.bytes,
            scales: &self.scales,
            global_scale: self.global_scale,
        }
    }
}

fn upload_linear<W: Nvfp4Shape, B: Nvfp4Shape>(
    stream: &CudaStream,
    linear: &LinearWeights<W, B>,
) -> AppResult<UploadedLinear> {
    Ok(UploadedLinear {
        weight: upload_nvfp4(stream, &linear.weight)?,
        bias: upload_nvfp4(stream, &linear.bias)?,
    })
}

fn upload_nvfp4<S: Nvfp4Shape>(
    stream: &CudaStream,
    tensor: &Nvfp4Tensor<S>,
) -> AppResult<UploadedNvfp4> {
    Ok(UploadedNvfp4 {
        bytes: DeviceBuffer::from_host(stream, tensor.bytes.as_ref())?,
        scales: DeviceBuffer::from_host(stream, tensor.scales.as_ref())?,
        global_scale: tensor.global_scale,
    })
}
