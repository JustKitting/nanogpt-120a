pub mod layer_norm;
pub mod nvfp4_quant;

pub const CUDA_OXIDE_PTX_PATH: &str = "rust_kernels_cuda.ptx";

pub mod ptx_assets {
    pub const NATIVE_NVFP4_MMA_DOWNCAST: &str =
        include_str!("ms_eden/native_nvfp4_mma_downcast.func.ptx");
}
