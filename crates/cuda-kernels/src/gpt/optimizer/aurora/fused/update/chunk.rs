use crate::amax::max4_f32;
use crate::f16_tc_matmul::cta_tile::CTA_THREADS;

use super::one::update_one;

macro_rules! update_at {
    ($u:expr, $z:expr, $x:expr, $rows:expr, $cols:expr, $len:expr, $transposed:expr,
     $scale:expr, $lr:expr, $wd:expr, $avg:expr, $base:expr, $tid:expr, $mul:expr) => {
        update_one(
            $u,
            $z,
            $x,
            $rows,
            $cols,
            $len,
            $transposed,
            $scale,
            $lr,
            $wd,
            $avg,
            $base + $tid + CTA_THREADS * $mul,
        )
    };
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update_four_amax(
    u: *const f32,
    z_master: *mut f32,
    x_master: *mut f32,
    rows: u32,
    cols: u32,
    len: u32,
    transposed: bool,
    scale: f32,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    base: u32,
    tid: u32,
) -> f32 {
    max4_f32(
        update_at!(
            u,
            z_master,
            x_master,
            rows,
            cols,
            len,
            transposed,
            scale,
            learning_rate,
            weight_decay,
            average_coefficient,
            base,
            tid,
            0
        ),
        update_at!(
            u,
            z_master,
            x_master,
            rows,
            cols,
            len,
            transposed,
            scale,
            learning_rate,
            weight_decay,
            average_coefficient,
            base,
            tid,
            1
        ),
        update_at!(
            u,
            z_master,
            x_master,
            rows,
            cols,
            len,
            transposed,
            scale,
            learning_rate,
            weight_decay,
            average_coefficient,
            base,
            tid,
            2
        ),
        update_at!(
            u,
            z_master,
            x_master,
            rows,
            cols,
            len,
            transposed,
            scale,
            learning_rate,
            weight_decay,
            average_coefficient,
            base,
            tid,
            3
        ),
    )
}
