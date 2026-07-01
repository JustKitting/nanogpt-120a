use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::{LayerNormTensors, LayerNormWeights, LinearWeights, Nvfp4Shape, Nvfp4Tensor};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

use crate::AppResult;

pub struct UploadedLinear {
    pub weight: UploadedNvfp4,
    pub bias: UploadedNvfp4,
}

pub struct UploadedLayerNorm {
    pub(crate) weight: UploadedNvfp4,
    pub(crate) bias: UploadedNvfp4,
}

impl UploadedLayerNorm {
    pub(super) fn new(stream: &CudaStream, layer_norm: &LayerNormWeights) -> AppResult<Self> {
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
    pub(crate) global_scale: DeviceBuffer<f32>,
    pub(crate) len: usize,
}

impl UploadedNvfp4 {
    pub(crate) fn from_host(
        stream: &CudaStream,
        bytes: &[u8],
        scales: &[u8],
        global_scale: f32,
        len: usize,
    ) -> AppResult<Self> {
        Ok(Self {
            bytes: DeviceBuffer::from_host(stream, bytes)?,
            scales: DeviceBuffer::from_host(stream, scales)?,
            global_scale: DeviceBuffer::from_host(stream, &[global_scale])?,
            len,
        })
    }

    pub(crate) fn global_scale_to_host(&self, stream: &CudaStream) -> AppResult<f32> {
        Ok(self.global_scale.to_host_vec(stream)?[0])
    }

    pub fn device(&self) -> Nvfp4DeviceTensor<'_> {
        Nvfp4DeviceTensor::new(&self.bytes, &self.scales, &self.global_scale)
    }

    pub fn mma(&self) -> Nvfp4FourSixMmaWeightTensor<'_> {
        Nvfp4FourSixMmaWeightTensor::new(&self.bytes, &self.scales, &self.global_scale)
    }
}

pub(super) fn upload_linear<W: Nvfp4Shape, B: Nvfp4Shape>(
    stream: &CudaStream,
    linear: &LinearWeights<W, B>,
) -> AppResult<UploadedLinear> {
    Ok(UploadedLinear {
        weight: upload_nvfp4(stream, &linear.weight)?,
        bias: upload_nvfp4(stream, &linear.bias)?,
    })
}

pub(super) fn upload_nvfp4<S: Nvfp4Shape>(
    stream: &CudaStream,
    tensor: &Nvfp4Tensor<S>,
) -> AppResult<UploadedNvfp4> {
    UploadedNvfp4::from_host(
        stream,
        tensor.bytes.as_ref(),
        tensor.scales.as_ref(),
        tensor.global_scale,
        tensor.len(),
    )
}
