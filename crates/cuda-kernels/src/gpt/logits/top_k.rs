use cuda_device::{DisjointSlice, SharedArray, thread};

use super::args::{LOGITS_TOP_K, LogitsTopKParams, TOPK_CANDIDATES, TOPK_THREADS_PER_BLOCK};
use super::ordering::better;

pub(super) fn logits_top_k_body(
    logits: &[f32],
    mut out_tokens: DisjointSlice<u32>,
    mut out_values: DisjointSlice<f32>,
    params: LogitsTopKParams,
    values: &mut SharedArray<f32, TOPK_CANDIDATES>,
    indices: &mut SharedArray<u32, TOPK_CANDIDATES>,
) {
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
        values[base + i] = local_values[i];
        indices[base + i] = local_indices[i];
        i += 1;
    }

    thread::sync_threads();

    if thread == 0 {
        let mut best_values = [f32::NEG_INFINITY; LOGITS_TOP_K];
        let mut best_indices = [u32::MAX; LOGITS_TOP_K];
        let mut candidate = 0;
        while candidate < TOPK_CANDIDATES {
            let value = values[candidate];
            let index = indices[candidate];
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
