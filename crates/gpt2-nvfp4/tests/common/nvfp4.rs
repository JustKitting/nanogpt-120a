#![allow(dead_code)]

use cuda_core::{CudaStream, DeviceBuffer, DriverError};

pub const E2M1_MIN_PAIR: u8 = 0x11;
pub const E2M1_ONE_PAIR: u8 = 0x22;
pub const E4M3_ONE: u8 = 0x38;

pub fn set_e2m1_one(bytes: &mut [u8], element: usize) {
    let byte = &mut bytes[element / 2];
    if element & 1 == 0 {
        *byte = (*byte & 0xf0) | 0x2;
    } else {
        *byte = (*byte & 0x0f) | 0x20;
    }
}

pub fn repeating_identity_bytes(byte_len: usize, cols: usize, row_len: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; byte_len];
    for col in 0..cols {
        set_e2m1_one(&mut bytes, col * row_len + col % row_len);
    }
    bytes
}

pub fn filled_u8(
    stream: &CudaStream,
    len: usize,
    value: u8,
) -> Result<DeviceBuffer<u8>, DriverError> {
    DeviceBuffer::from_host(stream, &vec![value; len])
}
