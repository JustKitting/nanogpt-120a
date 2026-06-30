use super::HADAMARD_DIM;

#[inline(always)]
pub(super) fn padded_chunk_position(chunk_base: u32, lane: u32, dst_row_len: u32) -> (u32, u32) {
    let row = chunk_base / dst_row_len;
    let row_base = row * dst_row_len;
    let chunk_in_row = chunk_base - row_base;
    (row, chunk_in_row + lane)
}

#[inline(always)]
pub(super) fn no_pad_pow2_chunk_position(
    chunk: u32,
    lane: u32,
    chunks_per_row_shift: u32,
) -> (u32, u32, bool) {
    let chunk_in_row_mask = (1u32 << chunks_per_row_shift) - 1;
    let row = chunk >> chunks_per_row_shift;
    let chunk_in_row = (chunk & chunk_in_row_mask) * HADAMARD_DIM;
    (row, chunk_in_row + lane, chunk_in_row == 0)
}

#[inline(always)]
pub(super) fn no_pad_chunk_position(
    chunk: u32,
    lane: u32,
    chunks_per_row: u32,
) -> (u32, u32, bool) {
    let row = chunk / chunks_per_row;
    let chunk_in_row = (chunk - row * chunks_per_row) * HADAMARD_DIM;
    (row, chunk_in_row + lane, chunk_in_row == 0)
}
