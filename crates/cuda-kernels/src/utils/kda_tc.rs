use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS};

macro_rules! tc_stage_loop {
    ($tile:expr, $a_tile:expr, $b_tile:expr, $acc:expr; $k_base:ident < $limit:expr;
     $stage_a:block $stage_b:block) => {{
        let mut $k_base = 0;
        while $k_base < $limit {
            $stage_a
            $stage_b
            thread::sync_threads();
            $crate::kda_tc::mma_accumulate($tile, $a_tile, $b_tile, &mut $acc);
            thread::sync_threads();
            $k_base += $crate::f16_tc_matmul::cta_tile::CTA_K;
        }
    }};
}

pub(crate) use tc_stage_loop;

macro_rules! for_acc_fragments {
    ($acc:expr, $tile:expr, |$warp_n:ident, $frag:ident, $value:ident| $body:block) => {{
        let mut i = 0;
        while i < 4 {
            let $warp_n = $tile.warp_n0 + i as u32;
            let mut $frag = 0;
            while $frag < 4 {
                let $value = $acc[i][$frag];
                $body
                $frag += 1;
            }
            i += 1;
        }
    }};
}

pub(crate) use for_acc_fragments;

pub(crate) type CtaATile = cuda_device::SharedArray<u16, CTA_A_ELEMS>;
pub(crate) type CtaBTile = cuda_device::SharedArray<u16, CTA_B_ELEMS>;

macro_rules! with_tc_ab_tiles {
    ($body:ident; $($arg:expr),* $(,)?) => { with_tc_ab_tiles!(@call $body; [$($arg),*]; []) };
    ($body:ident; $($arg:expr),* ; $($tail:expr),* $(,)?) => { with_tc_ab_tiles!(@call $body; [$($arg),*]; [$($tail),*]) };
    (@call $body:ident; [$($arg:expr),*]; [$($tail:expr),*]) => {{
        static mut A_TILE: $crate::kda_tc::CtaATile = cuda_device::SharedArray::UNINIT;
        static mut B_TILE: $crate::kda_tc::CtaBTile = cuda_device::SharedArray::UNINIT;
        $body($($arg,)* unsafe { &mut A_TILE }, unsafe { &mut B_TILE } $(, $tail)*);
    }};
}

pub(crate) use with_tc_ab_tiles;

#[path = "kda_tc/context.rs"]
mod context;
#[path = "kda_tc/stage.rs"]
mod stage;
#[path = "kda_tc/store.rs"]
mod store;

pub(crate) use {context::*, stage::*, store::*};
