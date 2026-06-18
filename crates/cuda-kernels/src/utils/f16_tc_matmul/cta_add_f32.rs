use cuda_device::{DisjointSlice, SharedArray, thread};

use super::cta_stage::{load_a_fragments, load_b_fragments};
use super::cta_stage_f32::stage_tiles_f32_b_t;
use super::cta_store_add::store_add;
use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS, CtaTile};
use crate::mma::mma_m16n8k16_f16_f16_f32;

#[allow(clippy::too_many_arguments)]
pub(super) fn cta_matmul_add_f32_body(
    a: &[f32],
    b_t: &[f32],
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
    let thread_id = thread::threadIdx_x();
    if thread_id >= CTA_THREADS || thread::blockIdx_z() >= batch_count {
        return;
    }

    let tile = CtaTile::new(thread_id);
    let mut acc0 = [0.0_f32; 4];
    let mut acc1 = [0.0_f32; 4];
    let mut acc2 = [0.0_f32; 4];
    let mut acc3 = [0.0_f32; 4];
    let mut k_base = 0;
    while k_base < k {
        stage_tiles_f32_b_t(a, b_t, a_tile, b_tile, tile, m, n, k, k_base);
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
        thread::sync_threads();
        k_base += CTA_K;
    }
    store_add(
        acc0,
        tile,
        tile.warp_n0,
        base,
        &mut out,
        m,
        n,
        base_scale,
        matmul_scale,
    );
    store_add(
        acc1,
        tile,
        tile.warp_n0 + 1,
        base,
        &mut out,
        m,
        n,
        base_scale,
        matmul_scale,
    );
    store_add(
        acc2,
        tile,
        tile.warp_n0 + 2,
        base,
        &mut out,
        m,
        n,
        base_scale,
        matmul_scale,
    );
    store_add(
        acc3,
        tile,
        tile.warp_n0 + 3,
        base,
        &mut out,
        m,
        n,
        base_scale,
        matmul_scale,
    );
}
