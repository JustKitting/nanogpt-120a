use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};

pub const ARGMAX_THREADS_PER_BLOCK: u32 = 256;
pub const TOPK_THREADS_PER_BLOCK: u32 = 128;
pub const LOGITS_TOP_K: usize = 32;
pub const TOPK_CANDIDATES: usize = TOPK_THREADS_PER_BLOCK as usize * LOGITS_TOP_K;
pub const WARP_SIZE: u32 = 32;
pub const ARGMAX_WARPS_PER_BLOCK: u32 = ARGMAX_THREADS_PER_BLOCK / WARP_SIZE;
pub const FULL_WARP_MASK: u32 = 0xffff_ffff;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LogitsArgmaxParams {
    pub row: u32,
    pub vocab_size: u32,
}

unsafe impl DeviceCopy for LogitsArgmaxParams {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LogitsTopKParams {
    pub row: u32,
    pub vocab_size: u32,
    pub k: u32,
}

unsafe impl DeviceCopy for LogitsTopKParams {}

pub struct LogitsArgmaxArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub logits: &'a DeviceBuffer<f32>,
    pub out_token: &'out mut DeviceBuffer<u32>,
    pub row: u32,
    pub vocab_size: u32,
}

pub struct LogitsTopKArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub logits: &'a DeviceBuffer<f32>,
    pub out_tokens: &'out mut DeviceBuffer<u32>,
    pub out_values: &'out mut DeviceBuffer<f32>,
    pub row: u32,
    pub vocab_size: u32,
    pub k: u32,
}
