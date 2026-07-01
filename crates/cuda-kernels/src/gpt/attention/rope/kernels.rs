use cuda_device::{DisjointSlice, cuda_module, kernel};

use super::ApplyRopeParams;
use super::body::{apply_rope_body, apply_rope_save_f16_body};

pub use module::{LoadedModule, from_module};

#[cuda_module]
pub mod module {
    use super::*;

    #[kernel]
    pub fn apply_rope_kernel(qkv: DisjointSlice<f32>, params: ApplyRopeParams) {
        apply_rope_body(qkv, params);
    }

    #[kernel]
    pub fn apply_rope_save_f16_kernel(
        qkv: DisjointSlice<f32>,
        qkv_f16: DisjointSlice<u16>,
        params: ApplyRopeParams,
    ) {
        apply_rope_save_f16_body(qkv, qkv_f16, params);
    }
}
