#[path = "gpt/attention.rs"]
pub mod attention;
#[path = "gpt/embedding.rs"]
pub mod embedding;
#[path = "utils/kernel_ops.rs"]
pub mod kernel_ops;
#[path = "gpt/layer_norm.rs"]
pub mod layer_norm;
#[path = "utils/nvfp4_quant.rs"]
pub mod nvfp4_quant;

pub mod gpt {
    pub use crate::{attention, embedding, layer_norm};
}

pub mod utils {
    pub use crate::{kernel_ops, nvfp4_quant};
}

pub const CUDA_OXIDE_PTX_PATH: &str = "rust_kernels_cuda.ptx";
