use std::{path::PathBuf, sync::Arc};

use cuda_core::{CudaContext, CudaModule, CudaStream, DriverError};

pub type CudaTestContext = (Arc<CudaContext>, Arc<CudaStream>, Arc<CudaModule>);

pub fn cuda_test_context() -> Result<CudaTestContext, DriverError> {
    let device_index = std::env::var("CUDA_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let ptx_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../rust_kernels_cuda.ptx")
        .to_string_lossy()
        .into_owned();
    let ctx = CudaContext::new(device_index)?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(ptx_path.as_str())?;
    Ok((ctx, stream, ptx))
}

#[allow(dead_code)]
pub fn assert_nonzero_finite(values: &[f32]) {
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| value.abs() > 0.0));
}
