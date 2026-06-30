use cuda_core::{CudaStream, DeviceBuffer};

use crate::AppResult;

const TRAIN_TRACE_ENV: &str = "TRAIN_TRACE";

pub fn enabled() -> bool {
    std::env::var(TRAIN_TRACE_ENV)
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

pub(super) fn f32_buffer_stats(
    stream: &CudaStream,
    buffer: &DeviceBuffer<f32>,
) -> AppResult<(f32, f32)> {
    let values = buffer.to_host_vec(stream)?;
    let mut sum_sq = 0.0f64;
    let mut max = 0.0f32;

    for value in &values {
        let abs = value.abs();
        sum_sq += (*value as f64) * (*value as f64);
        max = max.max(abs);
    }

    Ok(((sum_sq / values.len() as f64).sqrt() as f32, max))
}

pub(super) fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
