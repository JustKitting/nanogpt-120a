use super::super::rope::ApplyRopeParams;
use super::qkv::qkv_index_from_shape;

#[inline(always)]
pub(crate) fn rope_qkv_index(
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &ApplyRopeParams,
) -> usize {
    let row = batch * params.seq_len + token;
    qkv_index_from_shape(
        row,
        head,
        dim,
        section_offset,
        params.qkv_dim,
        params.head_dim,
    )
}
