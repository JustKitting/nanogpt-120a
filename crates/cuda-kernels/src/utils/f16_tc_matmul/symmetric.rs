use cuda_device::{DisjointSlice, thread};

use super::load::{load_a_fragments, load_b_fragments};
use super::tile::{K_STEP, Tile};
use crate::mma::mma_m16n8k16_f16_f16_f32;

pub(super) fn symmetric_matmul_body(x: &[u16], mut out: DisjointSlice<f32>, rows: u32, cols: u32) {
    let lane = thread::threadIdx_x();
    if lane >= 32 {
        return;
    }

    let tile = Tile::new(lane);
    if tile.col > tile.row + 15 {
        return;
    }

    let mut acc = [0.0_f32; 4];
    let mut k_base = 0;
    while k_base < cols {
        mma_m16n8k16_f16_f16_f32(
            load_a_fragments(x, 0, tile, k_base, rows, cols),
            load_b_fragments(x, 0, tile, k_base, rows, cols),
            &mut acc,
        );
        k_base += K_STEP;
    }

    store_symmetric(acc[0], tile, &mut out, rows, 0);
    store_symmetric(acc[1], tile, &mut out, rows, 1);
    store_symmetric(acc[2], tile, &mut out, rows, 2);
    store_symmetric(acc[3], tile, &mut out, rows, 3);
}

#[inline(always)]
fn store_symmetric(
    value: f32,
    tile: Tile,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    acc_index: usize,
) {
    let row = tile.row + tile.group + if acc_index < 2 { 0 } else { 8 };
    let col = tile.col + tile.thread_in_group * 2 + (acc_index as u32 & 1);
    if row < rows && col < rows && row >= col {
        unsafe {
            *out.get_unchecked_mut((row * rows + col) as usize) = value;
            if row != col {
                *out.get_unchecked_mut((col * rows + row) as usize) = value;
            }
        }
    }
}
