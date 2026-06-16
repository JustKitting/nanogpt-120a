#[path = "gpt/attention/mod.rs"]
pub mod attention;
#[path = "gpt/embedding.rs"]
pub mod embedding;
#[path = "utils/float_ptx.rs"]
pub mod float_ptx;
#[path = "gpt/layer_norm.rs"]
pub mod layer_norm;
#[path = "utils/mma/mod.rs"]
pub mod mma;
#[path = "utils/nvfp4/mod.rs"]
pub mod nvfp4;
#[path = "utils/nvfp4/cast.rs"]
pub mod nvfp4_cast;
#[path = "utils/nvfp4/quant/mod.rs"]
pub mod nvfp4_quant;
#[path = "utils/shuffle.rs"]
pub mod shuffle;
#[path = "utils/warp_reduce.rs"]
pub mod warp_reduce;

pub mod gpt {
    pub use crate::{attention, embedding, layer_norm};
}

pub mod utils {
    pub use crate::{float_ptx, mma, nvfp4, nvfp4_cast, nvfp4_quant, shuffle, warp_reduce};
}

pub const CUDA_OXIDE_PTX_PATH: &str = "rust_kernels_cuda.ptx";
