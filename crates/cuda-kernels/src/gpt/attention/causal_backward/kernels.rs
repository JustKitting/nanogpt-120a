use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};

use super::dkv::dkv_body;
use super::dq::dq_body;
use super::reductions::{HEAD_REDUCE_PAIR_LEN, KEY_REDUCE_PAIR_LEN};
use super::softmax_d::softmax_d_body;
use super::types::{
    CAUSAL_BACKWARD_HEAD_DIM_THREADS, CAUSAL_BACKWARD_KEY_BLOCK, CausalAttentionBackwardParams,
};

const DKV_THREADS: u32 = CAUSAL_BACKWARD_KEY_BLOCK * CAUSAL_BACKWARD_HEAD_DIM_THREADS;

#[allow(static_mut_refs)]
#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn softmax_d_kernel(
        out: &[f32],
        d_out: &[f32],
        softmax_d: DisjointSlice<f32>,
        params: CausalAttentionBackwardParams,
    ) {
        static mut REDUCE: SharedArray<f32, 2> = SharedArray::UNINIT;
        softmax_d_body(out, d_out, softmax_d, params, unsafe { &mut REDUCE });
    }

    #[kernel]
    pub fn dq_kernel(
        qkv: &[f32],
        d_out: &[f32],
        log_sum_exp: &[f32],
        softmax_d: &[f32],
        d_qkv: DisjointSlice<f32>,
        params: CausalAttentionBackwardParams,
    ) {
        static mut REDUCE: SharedArray<f32, HEAD_REDUCE_PAIR_LEN> = SharedArray::UNINIT;
        static mut DS: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut DQ_ROT: SharedArray<f32, { CAUSAL_BACKWARD_HEAD_DIM_THREADS as usize }> =
            SharedArray::UNINIT;
        dq_body(
            qkv,
            d_out,
            log_sum_exp,
            softmax_d,
            d_qkv,
            params,
            unsafe { &mut REDUCE },
            unsafe { &mut DS },
            unsafe { &mut DQ_ROT },
        );
    }

    #[kernel]
    pub fn dkv_kernel(
        qkv: &[f32],
        d_out: &[f32],
        log_sum_exp: &[f32],
        softmax_d: &[f32],
        d_qkv: DisjointSlice<f32>,
        params: CausalAttentionBackwardParams,
    ) {
        static mut REDUCE: SharedArray<f32, KEY_REDUCE_PAIR_LEN> = SharedArray::UNINIT;
        static mut PROB: SharedArray<f32, { CAUSAL_BACKWARD_KEY_BLOCK as usize }> =
            SharedArray::UNINIT;
        static mut DS: SharedArray<f32, { CAUSAL_BACKWARD_KEY_BLOCK as usize }> =
            SharedArray::UNINIT;
        static mut DK_ROT: SharedArray<f32, { DKV_THREADS as usize }> = SharedArray::UNINIT;
        dkv_body(
            qkv,
            d_out,
            log_sum_exp,
            softmax_d,
            d_qkv,
            params,
            unsafe { &mut REDUCE },
            unsafe { &mut PROB },
            unsafe { &mut DS },
            unsafe { &mut DK_ROT },
        );
    }
}

pub(crate) use module::{LoadedModule, from_module};
