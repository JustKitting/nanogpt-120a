use cuda_device::DisjointSlice;

use super::thread_index;
use crate::attention::CausalAttentionParams;
use crate::kda_common::{batch_head, chunk_count, chunk_matrix_elems};

pub(in super::super) fn mask_aqk_body(mut aqk: DisjointSlice<f32>, params: CausalAttentionParams) {
    mask_chunk_matrix(&mut aqk, params, false);
}

pub(in super::super) fn mask_akk_body(mut akk: DisjointSlice<f32>, params: CausalAttentionParams) {
    mask_chunk_matrix(&mut akk, params, true);
}

fn mask_chunk_matrix(matrix: &mut DisjointSlice<f32>, params: CausalAttentionParams, strict: bool) {
    let chunks = chunk_count(&params);
    let matrix_elems = chunk_matrix_elems(&params);
    let total = batch_head(&params) * chunks * matrix_elems;
    let Some(index) = thread_index(total) else {
        return;
    };
    let elem = index % matrix_elems;
    let row = elem / params.chunk_size;
    let col = elem - row * params.chunk_size;
    let keep = if strict { row > col } else { row >= col };
    if !keep {
        unsafe {
            *matrix.get_unchecked_mut(index as usize) = 0.0;
        }
    }
}
