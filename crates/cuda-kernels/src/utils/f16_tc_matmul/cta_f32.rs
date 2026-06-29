use cuda_device::{DisjointSlice, SharedArray, thread};

use super::cta_stage_f32::{stage_tiles_f32_b_t, stage_tiles_f32_b_t_aligned};
use super::cta_store::{store, store_aligned};
use super::cta_sync::sync_before_next_k;
use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_M, CTA_N, active_tile};

pub(super) fn cta_matmul_f32_body(
    a: &[f32],
    b_t: &[f32],
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
            stage_tiles_f32_b_t_aligned(a, b_t, a_tile, b_tile, tile, m, n, k, k_base);
        } else {
            stage_tiles_f32_b_t(a, b_t, a_tile, b_tile, tile, m, n, k, k_base);
        }
        thread::sync_threads();
        cta_mma4!(a_tile, b_tile, tile, acc0, acc1, acc2, acc3);
        sync_before_next_k(k_base, k);
        k_base += CTA_K;
    }
    if aligned {
        cta_store4!(store_aligned, tile, &mut out, m, n, acc0, acc1, acc2, acc3);
    } else {
        cta_store4!(store, tile, &mut out, m, n, acc0, acc1, acc2, acc3);
    }
}
