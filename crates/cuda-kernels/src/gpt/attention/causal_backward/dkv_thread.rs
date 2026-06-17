use super::types::CausalAttentionBackwardParams;

#[derive(Clone, Copy)]
pub(super) struct KeyThread {
    pub key_offset: u32,
    pub dim: u32,
    pub lane: u32,
    pub warp_in_key: u32,
    pub batch: u32,
    pub head: u32,
    pub block_key: u32,
}

impl KeyThread {
    #[inline(always)]
    pub(super) fn key(self) -> u32 {
        self.block_key + self.key_offset
    }

    #[inline(always)]
    pub(super) fn valid(self, params: &CausalAttentionBackwardParams) -> bool {
        let row = self.batch * params.seq_len + self.key();
        self.key() < params.seq_len && row < params.row_count && self.dim < params.head_dim
    }

    #[inline(always)]
    pub(super) fn active(self, query: u32, params: &CausalAttentionBackwardParams) -> bool {
        let query_row = self.batch * params.seq_len + query;
        self.valid(params)
            && query < params.seq_len
            && query_row < params.row_count
            && query >= self.key()
    }
}
