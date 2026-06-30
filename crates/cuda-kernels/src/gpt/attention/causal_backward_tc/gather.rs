use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;
use crate::attention::layout::{batched_qkv_index, compact_linear_parts, hidden_index, row_index};
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;

pub(super) const TC_BACKWARD_THREADS_PER_BLOCK: u32 = 256;

pub(super) fn gather_body(
    qkv: &[u16],
    d_out_src: &[f32],
    mut q: DisjointSlice<u16>,
    mut k: DisjointSlice<u16>,
    mut v: DisjointSlice<u16>,
    mut d_out: DisjointSlice<u16>,
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * TC_BACKWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
    let total = params.batch_size * params.head_count * params.seq_len * params.head_dim;
    if index >= total {
        return;
    }

    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let row = row_index(batch, token, &params);
    if row >= params.row_count {
        unsafe {
            *q.get_unchecked_mut(index as usize) = 0;
            *k.get_unchecked_mut(index as usize) = 0;
            *v.get_unchecked_mut(index as usize) = 0;
            *d_out.get_unchecked_mut(index as usize) = 0;
        }
        return;
    }

    unsafe {
        *q.get_unchecked_mut(index as usize) = qkv_value(qkv, batch, token, head, dim, 0, &params);
        *k.get_unchecked_mut(index as usize) =
            qkv_value(qkv, batch, token, head, dim, params.embedding_dim, &params);
        *v.get_unchecked_mut(index as usize) = qkv_value(
            qkv,
            batch,
            token,
            head,
            dim,
            params.embedding_dim * 2,
            &params,
        );
        *d_out.get_unchecked_mut(index as usize) =
            cvt_rn_f16_f32(d_out_src[hidden_index(batch, token, head, dim, &params)]);
    }
}

#[inline(always)]
fn qkv_value(
    qkv: &[u16],
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &CausalAttentionParams,
) -> u16 {
    qkv[batched_qkv_index(batch, token, head, dim, section_offset, params)]
}
