use cuda_device::{DisjointSlice, SharedArray, thread};

use super::cta_stage::{load_a_fragments, load_b_fragments, stage_tiles, stage_tiles_aligned};
use super::cta_store::{store, store_aligned};
use super::cta_sync::sync_before_next_k;
use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_M, CTA_N, active_tile};
use crate::mma::mma_m16n8k16_f16_f16_f32;

pub(super) fn cta_matmul_body(
    a: &[u16],
    b_t: &[u16],
    mut out: DisjointSlice<f32>,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
) {
    let Some(tile) = active_tile(batch_count) else {
        return;
    };
    let mut acc0 = [0.0_f32; 4];
    let mut acc1 = [0.0_f32; 4];
    let mut acc2 = [0.0_f32; 4];
    let mut acc3 = [0.0_f32; 4];
    let aligned = m % CTA_M == 0 && n % CTA_N == 0 && k % CTA_K == 0;
    let mut k_base = 0;
    while k_base < k {
        if aligned {
            stage_tiles_aligned(a, b_t, a_tile, b_tile, tile, m, n, k, k_base);
        } else {
            stage_tiles(a, b_t, a_tile, b_tile, tile, m, n, k, k_base);
        }
        thread::sync_threads();
        let a_fragments = load_a_fragments(a_tile, tile);
        mma_m16n8k16_f16_f16_f32(
            a_fragments,
            load_b_fragments(b_tile, tile, tile.warp_n0),
            &mut acc0,
        );
        mma_m16n8k16_f16_f16_f32(
            a_fragments,
            load_b_fragments(b_tile, tile, tile.warp_n0 + 1),
            &mut acc1,
        );
        mma_m16n8k16_f16_f16_f32(
            a_fragments,
            load_b_fragments(b_tile, tile, tile.warp_n0 + 2),
            &mut acc2,
        );
        mma_m16n8k16_f16_f16_f32(
            a_fragments,
            load_b_fragments(b_tile, tile, tile.warp_n0 + 3),
            &mut acc3,
        );
        sync_before_next_k(k_base, k);
        k_base += CTA_K;
    }
    if aligned {
        store_aligned(acc0, tile, tile.warp_n0, &mut out, m, n);
        store_aligned(acc1, tile, tile.warp_n0 + 1, &mut out, m, n);
        store_aligned(acc2, tile, tile.warp_n0 + 2, &mut out, m, n);
        store_aligned(acc3, tile, tile.warp_n0 + 3, &mut out, m, n);
    } else {
        store(acc0, tile, tile.warp_n0, &mut out, m, n);
        store(acc1, tile, tile.warp_n0 + 1, &mut out, m, n);
        store(acc2, tile, tile.warp_n0 + 2, &mut out, m, n);
        store(acc3, tile, tile.warp_n0 + 3, &mut out, m, n);
    }
}
