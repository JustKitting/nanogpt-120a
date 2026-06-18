use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use super::args::{
    ARGMAX_THREADS_PER_BLOCK, ARGMAX_WARPS_PER_BLOCK, FULL_WARP_MASK, LogitsArgmaxParams, WARP_SIZE,
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
}
