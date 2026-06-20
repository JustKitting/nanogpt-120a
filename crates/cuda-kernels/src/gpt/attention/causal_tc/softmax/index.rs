use crate::attention::CausalAttentionParams;

#[inline(always)]
pub(super) fn score(
    scores: &[f32],
    batch: u32,
    head: u32,
    query: u32,
    key: u32,
    params: &CausalAttentionParams,
) -> f32 {
    scores[score_index(batch, head, query, key, params)] * params.scale
}

#[inline(always)]
pub(super) fn score_index(
    batch: u32,
    head: u32,
    query: u32,
    key: u32,
    params: &CausalAttentionParams,
) -> usize {
    (((batch as usize * params.head_count as usize + head as usize) * params.seq_len as usize
        + query as usize)
        * params.seq_len as usize)
        + key as usize
}

#[inline(always)]
pub(super) fn log_sum_exp_index(
    batch: u32,
    token: u32,
    head: u32,
    params: &CausalAttentionParams,
) -> usize {
    (batch as usize * params.head_count as usize + head as usize) * params.seq_len as usize
        + token as usize
}
