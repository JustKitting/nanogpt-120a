macro_rules! cta_mma4 {
    ($a_tile:expr, $b_tile:expr, $tile:expr, $acc0:ident, $acc1:ident, $acc2:ident, $acc3:ident) => {{
        let tile = $tile;
        let a_fragments = $crate::f16_tc_matmul::cta_stage::load_a_fragments($a_tile, tile);
        $crate::mma::mma_m16n8k16_f16_f16_f32(
            a_fragments,
            $crate::f16_tc_matmul::cta_stage::load_b_fragments($b_tile, tile, tile.warp_n0),
            &mut $acc0,
        );
        $crate::mma::mma_m16n8k16_f16_f16_f32(
            a_fragments,
            $crate::f16_tc_matmul::cta_stage::load_b_fragments($b_tile, tile, tile.warp_n0 + 1),
            &mut $acc1,
        );
        $crate::mma::mma_m16n8k16_f16_f16_f32(
            a_fragments,
            $crate::f16_tc_matmul::cta_stage::load_b_fragments($b_tile, tile, tile.warp_n0 + 2),
            &mut $acc2,
        );
        $crate::mma::mma_m16n8k16_f16_f16_f32(
            a_fragments,
            $crate::f16_tc_matmul::cta_stage::load_b_fragments($b_tile, tile, tile.warp_n0 + 3),
            &mut $acc3,
        );
    }};
}

macro_rules! cta_store4 {
    ($store:path, $tile:expr, $out:expr, $m:expr, $n:expr, $acc0:ident, $acc1:ident, $acc2:ident, $acc3:ident) => {{
        let tile = $tile;
        $store($acc0, tile, tile.warp_n0, $out, $m, $n);
        $store($acc1, tile, tile.warp_n0 + 1, $out, $m, $n);
        $store($acc2, tile, tile.warp_n0 + 2, $out, $m, $n);
        $store($acc3, tile, tile.warp_n0 + 3, $out, $m, $n);
    }};
}

macro_rules! cta_store_add4 {
    (
        $store_add:path, $tile:expr, $base:expr, $out:expr, $m:expr, $n:expr, $base_scale:expr,
        $matmul_scale:expr, $($acc:ident => $offset:expr),+ $(,)?
    ) => {{
        let tile = $tile;
        $(
            $store_add(
                $acc,
                tile,
                tile.warp_n0 + $offset,
                $base,
                $out,
                $m,
                $n,
                $base_scale,
                $matmul_scale,
            );
        )+
    }};
}

mod args;
pub(crate) mod convert;
mod cta;
mod cta_add_f32;
mod cta_add_f32_rhs_transposed_base;
mod cta_f32;
mod cta_f32_a_transposed_half_rhs;
mod cta_f32_a_transposed_rhs;
mod cta_f32_half_rhs;
mod cta_f32_rhs;
pub(crate) mod cta_stage;
mod cta_stage_f32;
mod cta_stage_f32_transposed;
mod cta_store;
mod cta_store_add;
mod cta_sync;
pub(crate) mod cta_tile;
mod kernels;
mod launch_ops;
mod launcher;
mod launcher_add;
mod pad;
mod prepare;

pub use args::{
    F16ConvertArgs, F16TcMatmulAddArgs, F16TcMatmulAddRhsTransposeBaseArgs, F16TcMatmulArgs,
    F16TcMatmulF32ATransposedHalfRhsArgs, F16TcMatmulF32ATransposedRhsArgs, F16TcMatmulF32Args,
    F16TcMatmulF32HalfRhsArgs, F16TcMatmulF32RhsArgs, F16TcMatmulHalfArgs, F16TcMatmulScratch,
    f16_tc_matmul_elements, f16_tc_matmul_padded_k,
};
pub use launcher::F16TcMatmulModule;
