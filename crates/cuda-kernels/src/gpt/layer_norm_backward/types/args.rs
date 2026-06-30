use cuda_core::{CudaStream, DeviceBuffer};

use crate::nvfp4::Nvfp4DeviceTensor;

pub struct LayerNormBackwardInputArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub residual: &'a DeviceBuffer<u16>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
    pub weight: Nvfp4DeviceTensor<'a>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
}

pub struct LayerNormBackwardInputF32Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub residual: &'a DeviceBuffer<f32>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
    pub weight: Nvfp4DeviceTensor<'a>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
}

pub struct LayerNormBackwardParamArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub residual: &'a DeviceBuffer<u16>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
    pub d_weight: &'out mut DeviceBuffer<f32>,
    pub d_bias: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
}

pub struct LayerNormBackwardParamF32Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub residual: &'a DeviceBuffer<f32>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
    pub d_weight: &'out mut DeviceBuffer<f32>,
    pub d_bias: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
}
