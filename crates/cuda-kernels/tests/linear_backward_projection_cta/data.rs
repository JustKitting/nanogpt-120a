use cuda_core::{CudaStream, DeviceBuffer, DriverError};

const E4M3_ONE: u8 = 0x38;
const TOLERANCE: f32 = 1.0e-7;

pub fn upload_bytes(
    stream: &CudaStream,
    rows: usize,
    cols: usize,
    salt: usize,
) -> Result<DeviceBuffer<u8>, DriverError> {
    DeviceBuffer::from_host(stream, &patterned_e2m1(rows, cols, salt))
}

pub fn upload_scales(
    stream: &CudaStream,
    rows: usize,
    cols: usize,
) -> Result<DeviceBuffer<u8>, DriverError> {
    DeviceBuffer::from_host(stream, &vec![E4M3_ONE; rows * cols / 16])
}

pub fn row_scales(rows: usize, step: f32) -> Vec<f32> {
    (0..rows).map(|row| 1.0 + row as f32 * step).collect()
}

fn patterned_e2m1(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; rows * cols / 2];
    let values = [0x0, 0x1, 0x2, 0x3, 0xa, 0xb];
    for element in 0..rows * cols {
        set_nibble(&mut bytes, element, values[(element + salt) % values.len()]);
    }
    bytes
}

fn set_nibble(bytes: &mut [u8], element: usize, value: u8) {
    let byte = &mut bytes[element / 2];
    if element & 1 == 0 {
        *byte = (*byte & 0xf0) | value;
    } else {
        *byte = (*byte & 0x0f) | (value << 4);
    }
}

pub fn assert_vec_close(expected: &[f32], actual: &[f32]) {
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let error = (expected - actual).abs();
        assert!(
            error <= TOLERANCE,
            "index={index} expected={expected:.8e} actual={actual:.8e} error={error:.8e}"
        );
    }
}
