use cuda_core::{CudaStream, DeviceBuffer};

pub struct Nvfp4QuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub amax: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,
    pub out_global_scale: &'out mut DeviceBuffer<f32>,
    pub group_count: u32,
}

pub struct Nvfp4QuantRowwiseArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub amax: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,
    pub out_global_scale: &'out mut DeviceBuffer<f32>,
    pub group_count: u32,
    pub row_len: u32,
}

pub struct RowAmaxArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub row_len: u32,
}

pub struct MsEdenQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,
    pub out_global_scales: &'out mut DeviceBuffer<f32>,
    pub out_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub src_row_len: u32,
    pub dst_row_len: u32,
    pub global_scale: f32,
    pub scale_override: f32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub struct MsEdenDeviceScaleQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,
    pub out_global_scales: &'out mut DeviceBuffer<f32>,
    pub out_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub global_scale: &'a DeviceBuffer<f32>,
    pub row_count: u32,
    pub src_row_len: u32,
    pub dst_row_len: u32,
    pub scale_override: f32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub struct QuartetBackwardMsEdenQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,
    pub out_global_scales: &'out mut DeviceBuffer<f32>,
    pub out_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub src_row_len: u32,
    pub dst_row_len: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub struct QuartetBackwardMsEdenDeviceScaleQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,
    pub out_global_scales: &'out mut DeviceBuffer<f32>,
    pub out_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub out_global_scale: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub src_row_len: u32,
    pub dst_row_len: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}
