use cuda_core::{CudaStream, DeviceBuffer};

use crate::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};

pub struct Nvfp4DecodeTransposeArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: Nvfp4DeviceTensor<'a>,
    pub output: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub cols: u32,
}

pub struct Nvfp4RowwiseDecodeTransposeArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: Nvfp4RowwiseDeviceTensor<'a>,
    pub output: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub cols: u32,
}
