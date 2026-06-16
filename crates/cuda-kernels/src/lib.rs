pub mod attention;
pub mod embedding;
pub mod kernel_config;
pub mod layer_norm;
pub mod nvfp4_quant;

pub const CUDA_OXIDE_PTX_PATH: &str = "rust_kernels_cuda.ptx";
