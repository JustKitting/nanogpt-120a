use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};

use super::{CAUSAL_MAX_WARPS_PER_BLOCK, CausalAttentionParams};

#[path = "kernels/body.rs"]
mod body;
#[path = "kernels/score.rs"]
mod score;
use body::causal_attention_body;
use score::MAX_CAUSAL_TOKENS;

pub use module::{LoadedModule, from_module};

#[cuda_module]
pub mod module {
    use super::*;

    static mut SCORES: SharedArray<f32, MAX_CAUSAL_TOKENS> = SharedArray::UNINIT;
    static mut REDUCE: SharedArray<f32, { CAUSAL_MAX_WARPS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;

    #[kernel]
    pub fn causal_attention_kernel(
        qkv: &[f32],
        mut out: DisjointSlice<f32>,
        mut log_sum_exp: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        causal_attention_body(
            qkv,
            &mut out,
            &mut log_sum_exp,
            params,
            unsafe { &mut SCORES },
            unsafe { &mut REDUCE },
        );
    }
}
