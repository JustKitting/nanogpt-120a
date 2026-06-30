use cuda_core::LaunchConfig;

use super::config::{GROUPS_PER_BLOCK, THREADS_PER_BLOCK, WARPS_PER_BLOCK};
use super::kernels;
use crate::launch::grid_x_config;

const MS_EDEN_CHUNK_LEN: u32 = 32;

#[derive(Clone, Copy)]
pub(super) struct MsEdenPackGrid {
    pub chunk_count: u32,
    pub grid_dim: u32,
}

impl MsEdenPackGrid {
    pub fn for_elements(element_count: u32) -> Self {
        Self::from_chunks(element_count.div_ceil(MS_EDEN_CHUNK_LEN))
    }

    fn from_chunks(chunk_count: u32) -> Self {
        Self {
            chunk_count,
            grid_dim: chunk_count.div_ceil(WARPS_PER_BLOCK),
        }
    }

    pub fn config(self) -> LaunchConfig {
        grid_config(self.grid_dim)
    }

    pub fn is_exact(self) -> bool {
        self.chunk_count.is_multiple_of(WARPS_PER_BLOCK)
    }
}

#[derive(Clone, Copy)]
pub(super) struct Fp32PairNoPad {
    pub chunks_per_row: u32,
    pub transpose_chunks_per_row: u32,
}

impl Fp32PairNoPad {
    pub fn new(
        row_count: u32,
        src_row_len: u32,
        dst_row_len: u32,
        transpose_dst_row_len: u32,
    ) -> Option<Self> {
        (src_row_len == dst_row_len
            && row_count == transpose_dst_row_len
            && dst_row_len.is_multiple_of(MS_EDEN_CHUNK_LEN)
            && transpose_dst_row_len.is_multiple_of(MS_EDEN_CHUNK_LEN))
        .then_some(Self {
            chunks_per_row: dst_row_len / MS_EDEN_CHUNK_LEN,
            transpose_chunks_per_row: transpose_dst_row_len / MS_EDEN_CHUNK_LEN,
        })
    }

    pub fn pow2(self) -> Option<Fp32PairNoPadPow2> {
        (self.chunks_per_row.is_power_of_two() && self.transpose_chunks_per_row.is_power_of_two())
            .then_some(Fp32PairNoPadPow2 {
                chunks_per_row_shift: self.chunks_per_row.trailing_zeros(),
                transpose_chunks_per_row_shift: self.transpose_chunks_per_row.trailing_zeros(),
            })
    }
}

#[derive(Clone, Copy)]
pub(super) struct Fp32PairNoPadPow2 {
    pub chunks_per_row_shift: u32,
    pub transpose_chunks_per_row_shift: u32,
}

#[derive(Clone, Copy)]
pub(super) struct RowwiseTransposeNoPad {
    pub source_cols: u32,
    pub chunks_per_row_shift: u32,
}

impl RowwiseTransposeNoPad {
    pub fn new(source_rows: u32, source_cols: u32, dst_row_len: u32) -> Option<Self> {
        if source_rows != dst_row_len || !dst_row_len.is_multiple_of(MS_EDEN_CHUNK_LEN) {
            return None;
        }

        let chunks_per_row = dst_row_len / MS_EDEN_CHUNK_LEN;
        chunks_per_row.is_power_of_two().then_some(Self {
            source_cols,
            chunks_per_row_shift: chunks_per_row.trailing_zeros(),
        })
    }

    pub fn source_cols_shift(self) -> Option<u32> {
        self.source_cols
            .is_power_of_two()
            .then(|| self.source_cols.trailing_zeros())
    }
}

pub(super) fn grid_config(grid_x: u32) -> LaunchConfig {
    grid_x_config(grid_x, THREADS_PER_BLOCK)
}

pub(super) fn four_six_grid_config(group_count: u32) -> LaunchConfig {
    grid_config(group_count.div_ceil(GROUPS_PER_BLOCK))
}

pub(super) fn four_six_rowwise_pow2(row_len: u32, group_count: u32) -> bool {
    row_len.is_power_of_two() && group_count.is_multiple_of(GROUPS_PER_BLOCK)
}

pub(super) fn tensor_amax_chunk_count(element_count: u32) -> u32 {
    element_count.div_ceil(kernels::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK)
}
