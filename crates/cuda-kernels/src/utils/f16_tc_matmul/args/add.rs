use cuda_core::{CudaStream, DeviceBuffer};

use super::base::F16TcMatmulScratch;

pub struct F16TcMatmulAddArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub b_t: &'a DeviceBuffer<f32>,
    pub base: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub scratch: F16TcMatmulScratch<'scratch>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub base_scale: f32,
    pub matmul_scale: f32,
}

pub struct F16TcMatmulAddRhsTransposeBaseArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub rhs: &'a DeviceBuffer<f32>,
    pub base: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub base_scale: f32,
    pub matmul_scale: f32,
}
