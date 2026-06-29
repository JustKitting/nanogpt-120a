use cuda_device::{DisjointSlice, SharedArray, thread};

use super::cta_stage_f32::stage_tiles_f32_half_rhs;
use super::cta_store::store;
use super::cta_sync::sync_before_next_k;
use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, active_tile};

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn cta_matmul_f32_half_rhs_body(
    a: &[f32],
    rhs: &[u16],
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
    cta_accumulators!(acc0, acc1, acc2, acc3);
    let mut k_base = 0;
    while k_base < k {
        stage_tiles_f32_half_rhs(a, rhs, a_tile, b_tile, tile, m, n, k, k_base);
        thread::sync_threads();
        cta_mma4!(a_tile, b_tile, tile, acc0, acc1, acc2, acc3);
        sync_before_next_k(k_base, k);
        k_base += CTA_K;
    }
    cta_store4!(store, tile, &mut out, m, n, acc0, acc1, acc2, acc3);
}
