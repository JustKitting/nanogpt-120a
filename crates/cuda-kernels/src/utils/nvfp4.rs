use cuda_core::DeviceBuffer;

use crate::nvfp4_cast::{e2m1_value, e4m3_value};

#[derive(Clone, Copy)]
pub struct Nvfp4DeviceTensor<'a> {
    pub bytes: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub global_scale: f32,
}

#[derive(Clone, Copy)]
pub struct Nvfp4RowwiseDeviceTensor<'a> {
    pub bytes: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub global_scales: &'a DeviceBuffer<f32>,
}

#[inline(always)]
pub fn nvfp4_value(bytes: &[u8], scales: &[u8], global_scale: f32, index: usize) -> f32 {
    let byte = bytes[index / 2];
    let payload = if index & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };

    e2m1_value(payload) * e4m3_value(scales[index / 16] as u16) * global_scale
}

#[inline(always)]
pub fn nvfp4_rowwise_value(
    bytes: &[u8],
    scales: &[u8],
    global_scales: &[f32],
    row_len: usize,
    row: usize,
    col: usize,
) -> f32 {
    let index = row * row_len + col;
    nvfp4_value(bytes, scales, global_scales[row], index)
}
