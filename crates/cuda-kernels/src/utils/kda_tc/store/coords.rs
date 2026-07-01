use crate::f16_tc_matmul::cta_tile::CtaTile;

#[inline(always)]
pub(crate) fn compact_fragment_coords(tile: CtaTile, warp_n: u32, acc_index: usize) -> (u32, u32) {
    let token_in_chunk =
        tile.row_base + tile.warp_m * 16 + tile.group + if acc_index < 2 { 0 } else { 8 };
    let dim = tile.col_base + warp_n * 8 + tile.thread_in_group * 2 + (acc_index as u32 & 1);
    (token_in_chunk, dim)
}
