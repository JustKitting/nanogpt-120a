use super::tile::Tile;

#[inline(always)]
pub(super) fn load_a_fragments(
    halves: &[u16],
    batch: u32,
    tile: Tile,
    k_base: u32,
    rows: u32,
    cols: u32,
) -> [u32; 4] {
    [
        load_a_fragment(halves, batch, tile, k_base, rows, cols, 0),
        load_a_fragment(halves, batch, tile, k_base, rows, cols, 1),
        load_a_fragment(halves, batch, tile, k_base, rows, cols, 2),
        load_a_fragment(halves, batch, tile, k_base, rows, cols, 3),
    ]
}

#[inline(always)]
pub(super) fn load_b_fragments(
    halves: &[u16],
    batch: u32,
    tile: Tile,
    k_base: u32,
    rows: u32,
    cols: u32,
) -> [u32; 2] {
    [
        load_b_fragment(halves, batch, tile, k_base, rows, cols, 0),
        load_b_fragment(halves, batch, tile, k_base, rows, cols, 1),
    ]
}

#[inline(always)]
fn load_a_fragment(
    halves: &[u16],
    batch: u32,
    tile: Tile,
    k_base: u32,
    rows: u32,
    cols: u32,
    register: u32,
) -> u32 {
    let row = tile.row + tile.group + if register & 1 == 0 { 0 } else { 8 };
    let col = k_base + tile.thread_in_group * 2 + if register < 2 { 0 } else { 8 };
    if row < rows && col + 1 < cols {
        load_packed2(halves, ((batch * rows + row) * cols + col) as usize)
    } else {
        0
    }
}

#[inline(always)]
fn load_b_fragment(
    halves: &[u16],
    batch: u32,
    tile: Tile,
    k_base: u32,
    rows: u32,
    cols: u32,
    register: u32,
) -> u32 {
    let row = tile.col + tile.group;
    let col = k_base + tile.thread_in_group * 2 + if register == 0 { 0 } else { 8 };
    if row < rows && col + 1 < cols {
        load_packed2(halves, ((batch * rows + row) * cols + col) as usize)
    } else {
        0
    }
}

#[inline(always)]
fn load_packed2(halves: &[u16], base: usize) -> u32 {
    (halves[base] as u32) | ((halves[base + 1] as u32) << 16)
}
