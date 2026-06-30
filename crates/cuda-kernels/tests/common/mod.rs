use std::path::PathBuf;

pub fn gpu_device_index() -> usize {
    std::env::var("CUDA_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

pub fn ptx_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../rust_kernels_cuda.ptx")
        .to_string_lossy()
        .into_owned()
}

#[allow(dead_code)]
pub fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .fold(0.0, |max, (a, e)| max.max((a - e).abs()))
}

#[allow(dead_code)]
pub fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "actual={actual:.8e} expected={expected:.8e} error={error:.8e} tolerance={tolerance:.8e}"
    );
}
