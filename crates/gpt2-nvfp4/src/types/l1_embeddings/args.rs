use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::embedding::EmbeddingModule;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

pub struct TokenEmbeddingArgs<'a> {
    pub module: &'a EmbeddingModule,
    pub stream: &'a CudaStream,
    pub tokens: &'a DeviceBuffer<u32>,
    pub token_embedding: Nvfp4DeviceTensor<'a>,
    pub residual: &'a mut DeviceBuffer<f32>,
    pub normalized: &'a mut DeviceBuffer<f32>,
    pub normalized_amax: &'a mut DeviceBuffer<f32>,
    pub mean: &'a mut DeviceBuffer<f32>,
    pub inv_std: &'a mut DeviceBuffer<f32>,
}

pub struct HiddenStateDevice<'a> {
    pub stream: &'a CudaStream,
    pub residual: &'a mut DeviceBuffer<f32>,
    pub normalized: &'a mut DeviceBuffer<f32>,
    pub normalized_amax: &'a mut DeviceBuffer<f32>,
    pub mean: &'a mut DeviceBuffer<f32>,
    pub inv_std: &'a mut DeviceBuffer<f32>,
}
