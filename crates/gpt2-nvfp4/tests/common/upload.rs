use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::{Gpt2BlockWeights, LayerNormTensors, LayerNormWeights, Nvfp4Shape, Nvfp4Tensor};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub struct UploadedNvfp4 {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
}

impl UploadedNvfp4 {
    pub fn device(&self) -> Nvfp4DeviceTensor<'_> {
        Nvfp4DeviceTensor {
            bytes: &self.bytes,
            scales: &self.scales,
            global_scale: &self.global_scale,
        }
    }

    pub fn mma(&self) -> Nvfp4FourSixMmaWeightTensor<'_> {
        Nvfp4FourSixMmaWeightTensor {
            bytes: &self.bytes,
            scales: &self.scales,
            global_scale: &self.global_scale,
        }
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
        LayerNormTensors {
            weight: self.weight.device(),
            bias: self.bias.device(),
        }
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
    Ok(UploadedLayerNorm {
        weight: upload_nvfp4(stream, &layer_norm.weight)?,
        bias: upload_nvfp4(stream, &layer_norm.bias)?,
    })
}

pub fn upload_nvfp4<S: Nvfp4Shape>(
    stream: &CudaStream,
    tensor: &Nvfp4Tensor<S>,
) -> TestResult<UploadedNvfp4> {
    Ok(UploadedNvfp4 {
        bytes: DeviceBuffer::from_host(stream, tensor.bytes.as_ref())?,
        scales: DeviceBuffer::from_host(stream, tensor.scales.as_ref())?,
        global_scale: DeviceBuffer::from_host(stream, &[tensor.global_scale])?,
    })
}

fn upload_linear<W: Nvfp4Shape, B: Nvfp4Shape>(
    stream: &CudaStream,
    linear: &gpt2_nvfp4::LinearWeights<W, B>,
) -> TestResult<UploadedLinear> {
    Ok(UploadedLinear {
        weight: upload_nvfp4(stream, &linear.weight)?,
        bias: upload_nvfp4(stream, &linear.bias)?,
    })
}
