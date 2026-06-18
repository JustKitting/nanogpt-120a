use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use super::args::{
    ARGMAX_THREADS_PER_BLOCK, ARGMAX_WARPS_PER_BLOCK, FULL_WARP_MASK, LOGITS_TOP_K,
    LogitsArgmaxParams, LogitsTopKParams, TOPK_CANDIDATES, TOPK_THREADS_PER_BLOCK, WARP_SIZE,
};

#[allow(static_mut_refs)]
#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn logits_argmax_kernel(
        logits: &[f32],
        mut out_token: DisjointSlice<u32>,
        params: LogitsArgmaxParams,
    ) {
        static mut VALUES: SharedArray<f32, { ARGMAX_WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, { ARGMAX_WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / WARP_SIZE;
        let row_base = params.row as usize * params.vocab_size as usize;
        let mut best_value = f32::NEG_INFINITY;
        let mut best_index = u32::MAX;
        let mut col = thread;

        while col < params.vocab_size {
            let value = logits[row_base + col as usize];
            if better(value, col, best_value, best_index) {
                best_value = value;
                best_index = col;
            }
            col += ARGMAX_THREADS_PER_BLOCK;
        }

        let (warp_value, warp_index) = warp_argmax(best_value, best_index);
        if lane == 0 {
            unsafe {
                VALUES[warp_in_block as usize] = warp_value;
                INDICES[warp_in_block as usize] = warp_index;
            }
        }

        thread::sync_threads();

        if warp_in_block == 0 {
            let partial_value = if lane < ARGMAX_WARPS_PER_BLOCK {
                unsafe { VALUES[lane as usize] }
            } else {
                f32::NEG_INFINITY
            };
            let partial_index = if lane < ARGMAX_WARPS_PER_BLOCK {
                unsafe { INDICES[lane as usize] }
            } else {
                u32::MAX
            };
            let (_, index) = warp_argmax(partial_value, partial_index);
            if lane == 0 {
                unsafe {
                    *out_token.get_unchecked_mut(0) = index;
                }
            }
        }
    }

    #[inline(always)]
    fn warp_argmax(mut value: f32, mut index: u32) -> (f32, u32) {
        reduce_step(&mut value, &mut index, 16);
        reduce_step(&mut value, &mut index, 8);
        reduce_step(&mut value, &mut index, 4);
        reduce_step(&mut value, &mut index, 2);
        reduce_step(&mut value, &mut index, 1);
        (value, index)
    }

    #[inline(always)]
    fn reduce_step(value: &mut f32, index: &mut u32, lane_mask: u32) {
        let peer_value = warp::shuffle_xor_f32_sync(FULL_WARP_MASK, *value, lane_mask);
        let peer_index = warp::shuffle_xor_sync(FULL_WARP_MASK, *index, lane_mask);
        if better(peer_value, peer_index, *value, *index) {
            *value = peer_value;
            *index = peer_index;
        }
    }

    #[inline(always)]
    fn better(value: f32, index: u32, best_value: f32, best_index: u32) -> bool {
        value > best_value || (value == best_value && index < best_index)
    }

    #[kernel]
    pub fn logits_top_k_kernel(
        logits: &[f32],
        mut out_tokens: DisjointSlice<u32>,
        mut out_values: DisjointSlice<f32>,
        params: LogitsTopKParams,
    ) {
        static mut VALUES: SharedArray<f32, TOPK_CANDIDATES> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, TOPK_CANDIDATES> = SharedArray::UNINIT;

        let thread = thread::threadIdx_x();
        let row_base = params.row as usize * params.vocab_size as usize;
        let mut local_values = [f32::NEG_INFINITY; LOGITS_TOP_K];
        let mut local_indices = [u32::MAX; LOGITS_TOP_K];
        let mut col = thread;

        while col < params.vocab_size {
            insert_top_k(
                logits[row_base + col as usize],
                col,
                &mut local_values,
                &mut local_indices,
                params.k,
            );
            col += TOPK_THREADS_PER_BLOCK;
        }

        let base = thread as usize * LOGITS_TOP_K;
        let mut i = 0;
        while i < LOGITS_TOP_K {
            unsafe {
                VALUES[base + i] = local_values[i];
                INDICES[base + i] = local_indices[i];
            }
            i += 1;
        }

        thread::sync_threads();

        if thread == 0 {
            let mut best_values = [f32::NEG_INFINITY; LOGITS_TOP_K];
            let mut best_indices = [u32::MAX; LOGITS_TOP_K];
            let mut candidate = 0;
            while candidate < TOPK_CANDIDATES {
                let value = unsafe { VALUES[candidate] };
                let index = unsafe { INDICES[candidate] };
                insert_top_k(value, index, &mut best_values, &mut best_indices, params.k);
                candidate += 1;
            }

            let mut out = 0;
            while out < LOGITS_TOP_K && out < params.k as usize {
                unsafe {
                    *out_tokens.get_unchecked_mut(out) = best_indices[out];
                    *out_values.get_unchecked_mut(out) = best_values[out];
                }
                out += 1;
            }
        }
    }

    #[inline(always)]
    fn insert_top_k(
        value: f32,
        index: u32,
        values: &mut [f32; LOGITS_TOP_K],
        indices: &mut [u32; LOGITS_TOP_K],
        k: u32,
    ) {
        if index == u32::MAX || k == 0 {
            return;
        }

        let mut pos = 0;
        while pos < LOGITS_TOP_K && pos < k as usize {
            if better(value, index, values[pos], indices[pos]) {
                let mut shift = LOGITS_TOP_K - 1;
                while shift > pos {
                    values[shift] = values[shift - 1];
                    indices[shift] = indices[shift - 1];
                    shift -= 1;
                }
                values[pos] = value;
                indices[pos] = index;
                return;
            }
            pos += 1;
        }
    }
}
