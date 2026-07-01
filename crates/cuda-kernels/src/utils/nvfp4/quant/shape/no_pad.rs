use super::grid::ms_eden_chunks;

#[derive(Clone, Copy)]
pub(in crate::nvfp4_quant) struct Fp32PairNoPad {
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
        let chunks_per_row = ms_eden_chunks(dst_row_len)?;
        let transpose_chunks_per_row = ms_eden_chunks(transpose_dst_row_len)?;
        (src_row_len == dst_row_len && row_count == transpose_dst_row_len).then_some(Self {
            chunks_per_row,
            transpose_chunks_per_row,
        })
    }

    pub fn pow2_shifts(self) -> Option<(u32, u32)> {
        Some((
            pow2_shift(self.chunks_per_row)?,
            pow2_shift(self.transpose_chunks_per_row)?,
        ))
    }
}

#[derive(Clone, Copy)]
pub(in crate::nvfp4_quant) struct RowwiseTransposeNoPad {
    pub source_cols: u32,
    pub chunks_per_row_shift: u32,
}

impl RowwiseTransposeNoPad {
    pub fn new(source_rows: u32, source_cols: u32, dst_row_len: u32) -> Option<Self> {
        if source_rows != dst_row_len {
            return None;
        }

        let chunks_per_row = ms_eden_chunks(dst_row_len)?;
        Some(Self {
            source_cols,
            chunks_per_row_shift: pow2_shift(chunks_per_row)?,
        })
    }

    pub fn source_cols_shift(self) -> Option<u32> {
        pow2_shift(self.source_cols)
    }
}

fn pow2_shift(value: u32) -> Option<u32> {
    value.is_power_of_two().then(|| value.trailing_zeros())
}
