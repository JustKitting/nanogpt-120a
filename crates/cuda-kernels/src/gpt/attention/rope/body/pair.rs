use cuda_device::DisjointSlice;

use super::super::ApplyRopeParams;
use crate::attention::layout::rope_qkv_index;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;

#[inline(always)]
pub(super) fn read_pair(
    qkv: &mut DisjointSlice<f32>,
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &ApplyRopeParams,
) -> QkvPair {
    let even_index = rope_qkv_index(batch, token, head, dim, section_offset, params);
    let odd_index = rope_qkv_index(batch, token, head, dim + 1, section_offset, params);
    let ptr = qkv.as_mut_ptr();

    QkvPair {
        even_index,
        even: unsafe { *ptr.add(even_index) },
        odd_index,
        odd: unsafe { *ptr.add(odd_index) },
    }
}

#[inline(always)]
pub(super) fn store_pair_f16(qkv_f16: &mut DisjointSlice<u16>, pair: QkvPair) {
    unsafe {
        *qkv_f16.get_unchecked_mut(pair.even_index) = cvt_rn_f16_f32(pair.even);
        *qkv_f16.get_unchecked_mut(pair.odd_index) = cvt_rn_f16_f32(pair.odd);
    }
}

#[derive(Clone, Copy)]
pub(super) struct QkvPair {
    pub(super) even_index: usize,
    pub(super) even: f32,
    pub(super) odd_index: usize,
    pub(super) odd: f32,
}
