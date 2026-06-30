use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::float_ptx::abs_f32;
use crate::layer_norm_reduce::layer_norm_block_reduce;
use crate::warp_reduce::{thread_lane_warp, warp_sum_f32};

use super::super::args::NextLatShape;

pub(super) const THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
pub(super) const WARPS_PER_BLOCK: u32 = THREADS_PER_BLOCK / WARP_SIZE;

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

pub(super) fn nextlat_smooth_l1_body(
    predicted_next_states: &[f32],
    target_states: &[f32],
    losses: &mut DisjointSlice<f32>,
    d_predicted_next_states: &mut DisjointSlice<f32>,
    shape: NextLatShape,
) {
    static mut SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

    let batch = thread::blockIdx_x();
    let pos = thread::blockIdx_y();
    let row = batch * shape.seq_len + pos;
    let valid = batch < shape.batch_size && pos + 1 < shape.seq_len;
    let (thread, lane, warp_in_block) = thread_lane_warp();
    let valid_rows = shape.batch_size * (shape.seq_len - 1);
    let grad_scale = shape.lambda / (valid_rows * shape.embedding_dim) as f32;

    let mut local = 0.0;
    let mut col = thread;
    while col < shape.embedding_dim {
        let offset = (row * shape.embedding_dim + col) as usize;
        if valid {
            let target_offset = ((row + 1) * shape.embedding_dim + col) as usize;
            let diff = predicted_next_states[offset] - target_states[target_offset];
            let abs = abs_f32(diff);
            let grad = if abs < 1.0 {
                local += 0.5 * diff * diff;
                diff
            } else {
                local += abs - 0.5;
                if diff < 0.0 { -1.0 } else { 1.0 }
            };
            unsafe {
                *d_predicted_next_states.get_unchecked_mut(offset) = grad * grad_scale;
            }
        } else {
            unsafe {
                *d_predicted_next_states.get_unchecked_mut(offset) = 0.0;
            }
        }
        col += THREADS_PER_BLOCK;
    }

    let sum = layer_norm_block_reduce!(
        SUMS,
        WARPS_PER_BLOCK,
        local,
        lane,
        warp_in_block,
        warp_sum_f32
    );
    let row_loss = if valid {
        shape.lambda * sum / shape.embedding_dim as f32
    } else {
        0.0
    };
    if warp_in_block == 0 && lane == 0 {
        unsafe {
            *losses.get_unchecked_mut(row as usize) = row_loss;
        }
    }
}
