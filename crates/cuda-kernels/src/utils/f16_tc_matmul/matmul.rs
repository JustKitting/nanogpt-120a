use cuda_device::{DisjointSlice, thread};

use super::load::{load_a_fragments, load_b_fragments};
use super::store::store;
use super::tile::{K_STEP, Tile};
use crate::mma::mma_m16n8k16_f16_f16_f32;

pub(super) fn matmul_body(
    a: &[u16],
    b_t: &[u16],
    mut out: DisjointSlice<f32>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
) {
    let lane = thread::threadIdx_x();
    if lane >= 32 || thread::blockIdx_z() >= batch_count {
        return;
    }
    let tile = Tile::new(lane);
    let batch = thread::blockIdx_z();
    let mut acc = [0.0_f32; 4];
    let mut k_base = 0;
    while k_base < k {
        mma_m16n8k16_f16_f16_f32(
            load_a_fragments(a, batch, tile, k_base, m, k),
            load_b_fragments(b_t, batch, tile, k_base, n, k),
            &mut acc,
        );
        k_base += K_STEP;
    }
    store(acc, batch, tile, &mut out, m, n);
}
