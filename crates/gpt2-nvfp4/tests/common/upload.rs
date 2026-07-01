#![allow(dead_code)]

use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::{
    AttentionProjectionTensors, Gpt2BlockWeights, LayerNormTensors, LayerNormWeights,
    MlpDownTensors, MlpProjectionTensors, MlpUpTensors, Nvfp4Shape, Nvfp4Tensor,
};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

use crate::common::nvfp4::E4M3_ONE;

pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub struct UploadedNvfp4 {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
}

impl UploadedNvfp4 {
    pub fn device(&self) -> Nvfp4DeviceTensor<'_> {
        Nvfp4DeviceTensor::new(&self.bytes, &self.scales, &self.global_scale)
    }

    pub fn mma(&self) -> Nvfp4FourSixMmaWeightTensor<'_> {
        Nvfp4FourSixMmaWeightTensor::new(&self.bytes, &self.scales, &self.global_scale)
    }
}

pub fn attention_projection_tensors<'a>(
    qkv_weight: &'a UploadedNvfp4,
    qkv_bias: &'a UploadedNvfp4,
    c_proj_weight: &'a UploadedNvfp4,
    c_proj_bias: &'a UploadedNvfp4,
) -> AttentionProjectionTensors<'a> {
    AttentionProjectionTensors {
        qkv_weight: qkv_weight.mma(), qkv_bias: qkv_bias.device(),
        c_proj_weight: c_proj_weight.mma(), c_proj_bias: c_proj_bias.device(),
    }
}

pub fn mlp_projection_tensors<'a>(
    up_weight: &'a UploadedNvfp4,
    up_bias: &'a UploadedNvfp4,
    down_weight: &'a UploadedNvfp4,
    down_bias: &'a UploadedNvfp4,
) -> MlpProjectionTensors<'a> {
    MlpProjectionTensors {
        up: MlpUpTensors { weight: up_weight.mma(), bias: up_bias.device() },
        down: MlpDownTensors { weight: down_weight.mma(), bias: down_bias.device() },
    }
}

pub struct UploadedPair {
    pub weight: UploadedNvfp4,
    pub bias: UploadedNvfp4,
}

pub type UploadedLayerNorm = UploadedPair;
pub type UploadedLinear = UploadedPair;

impl UploadedPair {
    pub fn tensors(&self) -> LayerNormTensors<'_> {
        LayerNormTensors { weight: self.weight.device(), bias: self.bias.device() }
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

pub fn upload_layer_norm(
    stream: &CudaStream,
    layer_norm: &LayerNormWeights,
) -> TestResult<UploadedLayerNorm> {
    Ok(uploaded_pair(upload_nvfp4(stream, &layer_norm.weight)?, upload_nvfp4(stream, &layer_norm.bias)?))
}

pub fn upload_nvfp4<S: Nvfp4Shape>(
    stream: &CudaStream,
    tensor: &Nvfp4Tensor<S>,
) -> TestResult<UploadedNvfp4> {
    upload_nvfp4_parts(stream, tensor.bytes.as_ref(), tensor.scales.as_ref(), tensor.global_scale)
}

pub fn upload_nvfp4_bytes<S: Nvfp4Shape>(
    stream: &CudaStream,
    bytes: Vec<u8>,
) -> TestResult<UploadedNvfp4> {
    assert_eq!(bytes.len(), S::BYTE_LEN);
    upload_nvfp4_parts(stream, &bytes, &vec![E4M3_ONE; S::SCALE_LEN], 1.0)
}

fn upload_nvfp4_parts(stream: &CudaStream, bytes: &[u8], scales: &[u8], global_scale: f32) -> TestResult<UploadedNvfp4> {
    Ok(UploadedNvfp4 {
        bytes: DeviceBuffer::from_host(stream, bytes)?,
        scales: DeviceBuffer::from_host(stream, scales)?,
        global_scale: DeviceBuffer::from_host(stream, &[global_scale])?,
    })
}

pub fn upload_zero_nvfp4<S: Nvfp4Shape>(stream: &CudaStream) -> TestResult<UploadedNvfp4> {
    upload_nvfp4_bytes::<S>(stream, vec![0; S::BYTE_LEN])
}

fn upload_linear<W: Nvfp4Shape, B: Nvfp4Shape>(
    stream: &CudaStream,
    linear: &gpt2_nvfp4::LinearWeights<W, B>,
) -> TestResult<UploadedLinear> {
    Ok(uploaded_pair(upload_nvfp4(stream, &linear.weight)?, upload_nvfp4(stream, &linear.bias)?))
}

fn uploaded_pair(weight: UploadedNvfp4, bias: UploadedNvfp4) -> UploadedPair {
    UploadedPair { weight, bias }
}
