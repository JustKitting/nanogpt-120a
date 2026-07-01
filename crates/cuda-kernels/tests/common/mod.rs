#![allow(dead_code)]

use std::{path::PathBuf, sync::Arc};

use cuda_core::{CudaContext, CudaModule, CudaStream, DriverError};

pub mod f16_tc;
pub mod nvfp4;
pub mod nvfp4_tc;

pub type CudaTestContext = (Arc<CudaContext>, Arc<CudaStream>, Arc<CudaModule>);

pub fn gpu_device_index() -> usize {
    std::env::var("CUDA_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn ptx_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../rust_kernels_cuda.ptx")
        .to_string_lossy()
        .into_owned()
}

pub fn cuda_test_context() -> Result<CudaTestContext, DriverError> {
    let ctx = CudaContext::new(gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(ptx_path().as_str())?;
    Ok((ctx, stream, ptx))
}

pub fn cuda_test_module<M>(
    load: impl FnOnce(Arc<CudaModule>) -> Result<M, DriverError>,
) -> Result<(Arc<CudaContext>, Arc<CudaStream>, M), DriverError> {
    let (ctx, stream, ptx) = cuda_test_context()?;
    Ok((ctx, stream, load(ptx)?))
}

pub fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "actual={actual:.8e} expected={expected:.8e} error={error:.8e} tolerance={tolerance:.8e}"
    );
}

pub fn assert_all_close(actual: &[f32], expected: f32, tolerance: f32) {
    for actual in actual {
        assert_close(*actual, expected, tolerance);
    }
}

pub fn assert_slice_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "index={index} actual={actual:.8e} expected={expected:.8e} error={error:.8e} tolerance={tolerance:.8e}"
        );
    }
}
