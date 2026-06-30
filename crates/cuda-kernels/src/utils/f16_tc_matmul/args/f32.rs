use cuda_core::{CudaStream, DeviceBuffer};

pub struct F16TcMatmulF32Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub b_t: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub struct F16TcMatmulF32RhsArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub rhs: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub struct F16TcMatmulF32HalfRhsArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub rhs: &'a DeviceBuffer<u16>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub struct F16TcMatmulF32ATransposedRhsArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub rhs: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub struct F16TcMatmulF32ATransposedHalfRhsArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub rhs: &'a DeviceBuffer<u16>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub batch_count: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}
