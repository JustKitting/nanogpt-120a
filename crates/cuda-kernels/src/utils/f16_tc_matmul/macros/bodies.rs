macro_rules! cta_bt_matmul_body_fn {
    ($name:ident, $lhs_ty:ty, $rhs_ty:ty, $stage:path, $stage_aligned:path) => {
        #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
        pub(super) fn $name(
            a: &[$lhs_ty], b_t: &[$rhs_ty],
            mut out: cuda_device::DisjointSlice<f32>,
            a_tile: &mut $crate::f16_tc_matmul::CtaATile,
            b_tile: &mut $crate::f16_tc_matmul::CtaBTile,
            batch_count: u32, m: u32, n: u32, k: u32,
        ) {
            let Some(tile) = $crate::f16_tc_matmul::cta_tile::active_tile(batch_count) else {
                return;
            };
            cta_accumulators!(acc0, acc1, acc2, acc3);
            let aligned = m.is_multiple_of($crate::f16_tc_matmul::cta_tile::CTA_M)
                && n.is_multiple_of($crate::f16_tc_matmul::cta_tile::CTA_N)
                && k.is_multiple_of($crate::f16_tc_matmul::cta_tile::CTA_K);
            let mut k_base = 0;
            while k_base < k {
                if aligned {
                    $stage_aligned(a, b_t, a_tile, b_tile, tile, m, n, k, k_base);
                } else {
                    $stage(a, b_t, a_tile, b_tile, tile, m, n, k, k_base);
                }
                cuda_device::thread::sync_threads();
                cta_mma4!(a_tile, b_tile, tile, acc0, acc1, acc2, acc3);
                $crate::f16_tc_matmul::cta_sync::sync_before_next_k(k_base, k);
                k_base += $crate::f16_tc_matmul::cta_tile::CTA_K;
            }
            if aligned {
                cta_store4!(
                    $crate::f16_tc_matmul::cta_store::store_aligned,
                    tile,
                    &mut out,
                    m,
                    n,
                    acc0,
                    acc1,
                    acc2,
                    acc3
                );
            } else {
                cta_store4!(
                    $crate::f16_tc_matmul::cta_store::store,
                    tile,
                    &mut out,
                    m,
                    n,
                    acc0,
                    acc1,
                    acc2,
                    acc3
                );
            }
        }
    };
}

macro_rules! cta_rhs_matmul_body_fn {
    ($name:ident, $rhs:ident: $rhs_ty:ty, $stage:path) => {
        #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
        pub(super) fn $name(
            a: &[f32], $rhs: &[$rhs_ty],
            mut out: cuda_device::DisjointSlice<f32>,
            a_tile: &mut $crate::f16_tc_matmul::CtaATile,
            b_tile: &mut $crate::f16_tc_matmul::CtaBTile,
            batch_count: u32, m: u32, n: u32, k: u32,
        ) {
            let Some(tile) = $crate::f16_tc_matmul::cta_tile::active_tile(batch_count) else {
                return;
            };
            cta_accumulators!(acc0, acc1, acc2, acc3);
            let mut k_base = 0;
            while k_base < k {
                $stage(a, $rhs, a_tile, b_tile, tile, m, n, k, k_base);
                cuda_device::thread::sync_threads();
                cta_mma4!(a_tile, b_tile, tile, acc0, acc1, acc2, acc3);
                $crate::f16_tc_matmul::cta_sync::sync_before_next_k(k_base, k);
                k_base += $crate::f16_tc_matmul::cta_tile::CTA_K;
            }
            cta_store4!(
                $crate::f16_tc_matmul::cta_store::store,
                tile,
                &mut out,
                m,
                n,
                acc0,
                acc1,
                acc2,
                acc3
            );
        }
    };
}

macro_rules! cta_add_matmul_body_fn {
    ($name:ident, $rhs:ident, $stage:path) => {
        #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
        pub(super) fn $name(
            a: &[f32], $rhs: &[f32], base: &[f32],
            mut out: cuda_device::DisjointSlice<f32>,
            a_tile: &mut $crate::f16_tc_matmul::CtaATile,
            b_tile: &mut $crate::f16_tc_matmul::CtaBTile,
            batch_count: u32, m: u32, n: u32, k: u32, base_scale: f32, matmul_scale: f32,
        ) {
            let Some(tile) = $crate::f16_tc_matmul::cta_tile::active_tile(batch_count) else {
                return;
            };
            cta_accumulators!(acc0, acc1, acc2, acc3);
            let mut k_base = 0;
            while k_base < k {
                $stage(a, $rhs, a_tile, b_tile, tile, m, n, k, k_base);
                cuda_device::thread::sync_threads();
                cta_mma4!(a_tile, b_tile, tile, acc0, acc1, acc2, acc3);
                $crate::f16_tc_matmul::cta_sync::sync_before_next_k(k_base, k);
                k_base += $crate::f16_tc_matmul::cta_tile::CTA_K;
            }
            cta_store_add4!(
                $crate::f16_tc_matmul::cta_store_add::store_add,
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
    };
}
