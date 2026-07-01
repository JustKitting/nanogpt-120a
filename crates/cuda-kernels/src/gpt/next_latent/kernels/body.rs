#[path = "body/smooth_l1.rs"]
mod smooth_l1;

use cuda_device::{DisjointSlice, thread};

use super::super::args::NextLatShape;

pub(super) const THREADS_PER_BLOCK: u32 = 256;
pub(super) use smooth_l1::nextlat_smooth_l1_body;

pub(super) fn nextlat_concat_input_body(
    next_token_embeddings: &[f32],
    current_states: &[f32],
    out: &mut DisjointSlice<f32>,
    shape: NextLatShape,
) {
    let row = thread::blockIdx_x();
    let thread = thread::threadIdx_x();
    if row < shape.row_count {
        let row_base = row as usize * shape.embedding_dim as usize;
        let out_base = row as usize * (shape.embedding_dim as usize * 2);
        let mut col = thread;
        while col < shape.embedding_dim {
            let col_index = col as usize;
            unsafe {
                *out.get_unchecked_mut(out_base + col_index) =
                    next_token_embeddings[row_base + col_index];
                *out.get_unchecked_mut(out_base + shape.embedding_dim as usize + col_index) =
                    current_states[row_base + col_index];
            }
            col += THREADS_PER_BLOCK;
        }
    }
}

pub(super) fn nextlat_concat_backward_body(
    d_concat: &[f32],
    d_predicted: &[f32],
    d_next_token_embeddings: &mut DisjointSlice<f32>,
    d_current_states: &mut DisjointSlice<f32>,
    shape: NextLatShape,
) {
    let row = thread::blockIdx_x();
    let thread = thread::threadIdx_x();
    if row < shape.row_count {
        let row_base = row as usize * shape.embedding_dim as usize;
        let concat_base = row as usize * (shape.embedding_dim as usize * 2);
        let mut col = thread;
        while col < shape.embedding_dim {
            let col_index = col as usize;
            unsafe {
                *d_next_token_embeddings.get_unchecked_mut(row_base + col_index) =
                    d_concat[concat_base + col_index];
                *d_current_states.get_unchecked_mut(row_base + col_index) = d_concat
                    [concat_base + shape.embedding_dim as usize + col_index]
                    + d_predicted[row_base + col_index];
            }
            col += THREADS_PER_BLOCK;
        }
    }
}
