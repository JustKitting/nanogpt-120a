use cuda_device::{DisjointSlice, SharedArray, thread};

use super::cta_stage_f32::stage_tiles_f32_rhs_transposed;
use super::cta_store_add::store_add;
use super::cta_sync::sync_before_next_k;
use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, active_tile};

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn cta_matmul_add_f32_rhs_transposed_base_body(
    a: &[f32],
    rhs: &[f32],
    base: &[f32],
    mut out: DisjointSlice<f32>,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
    base_scale: f32,
    matmul_scale: f32,
) {
    let Some(tile) = active_tile(batch_count) else {
        return;
    };
    cta_accumulators!(acc0, acc1, acc2, acc3);
    let mut k_base = 0;
    while k_base < k {
        stage_tiles_f32_rhs_transposed(a, rhs, a_tile, b_tile, tile, m, n, k, k_base);
        thread::sync_threads();
        cta_mma4!(a_tile, b_tile, tile, acc0, acc1, acc2, acc3);
        sync_before_next_k(k_base, k);
        k_base += CTA_K;
    }
    cta_store_add4!(
        store_add,
        tile,
        base,
        &mut out,
        m,
        n,
        base_scale,
        matmul_scale,
        acc0 => 0,
        acc1 => 1,
        acc2 => 2,
        acc3 => 3,
    );
}
