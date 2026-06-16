use std::path::PathBuf;

pub const GPU_DEVICE_INDEX: usize = 1;

pub fn ptx_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../rust_kernels_cuda.ptx")
        .to_string_lossy()
        .into_owned()
}
