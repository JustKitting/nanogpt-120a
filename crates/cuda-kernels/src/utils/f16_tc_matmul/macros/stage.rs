macro_rules! cta_stage_transposed_rhs_fn {
    ($name:ident, $rhs_ty:ty, |$rhs:ident, $index:ident| $value:expr) => {
        fn $name(
            $rhs: &[$rhs_ty],
            b_tile: &mut $crate::f16_tc_matmul::CtaBTile,
            tile: $crate::f16_tc_matmul::cta_tile::CtaTile,
            n: u32,
            k: u32,
            k_base: u32,
        ) {
            let mut offset = cuda_device::thread::threadIdx_x();
            while offset < $crate::f16_tc_matmul::cta_tile::CTA_B_ELEMS as u32 {
                let (global_row, global_col) =
                    $crate::f16_tc_matmul::cta_stage::stage_coords(offset, tile.col_base, k_base);
                b_tile[offset as usize] = if global_row < n && global_col < k {
                    let $index = ((tile.batch * k + global_col) * n + global_row) as usize;
                    $value
                } else {
                    0
                };
                offset += $crate::f16_tc_matmul::cta_tile::CTA_THREADS;
            }
        }
    };
}
