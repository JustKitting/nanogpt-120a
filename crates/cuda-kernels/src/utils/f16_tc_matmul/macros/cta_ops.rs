macro_rules! cta_accumulators {
    ($($acc:ident),+ $(,)?) => {
        $(let mut $acc = [0.0_f32; 4];)+
    };
}

macro_rules! cta_mma4 {
    ($a_tile:expr, $b_tile:expr, $tile:expr, $acc0:ident, $acc1:ident, $acc2:ident, $acc3:ident) => {
        cta_mma4!($a_tile, $b_tile, $tile, $acc0 => 0, $acc1 => 1, $acc2 => 2, $acc3 => 3)
    };
    ($a_tile:expr, $b_tile:expr, $tile:expr, $($acc:ident => $offset:expr),+ $(,)?) => {{
        let tile = $tile;
        let a_fragments = $crate::f16_tc_matmul::cta_stage::load_a_fragments($a_tile, tile);
        $(
            $crate::mma::mma_m16n8k16_f16_f16_f32(
                a_fragments,
                $crate::f16_tc_matmul::cta_stage::load_b_fragments($b_tile, tile, tile.warp_n0 + $offset),
                &mut $acc,
            );
        )+
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
