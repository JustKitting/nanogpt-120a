use cuda_device::thread;

use super::super::{ApplyRopeParams, THREADS_PER_BLOCK};

#[inline(always)]
pub(super) fn rope_position(params: &ApplyRopeParams) -> Option<(u32, u32, u32, u32)> {
    let half_head_dim = params.head_dim / 2;
    let index = thread::blockIdx_x() * THREADS_PER_BLOCK + thread::threadIdx_x();
    let total = params.batch_size * params.seq_len * params.head_count * half_head_dim;
    if index >= total {
        return None;
    }

    let pair = index % half_head_dim;
    let head = (index / half_head_dim) % params.head_count;
    let token = (index / (half_head_dim * params.head_count)) % params.seq_len;
    let batch = index / (half_head_dim * params.head_count * params.seq_len);
    if batch * params.seq_len + token >= params.row_count {
        return None;
    }

    Some((batch, token, head, pair * 2))
}
