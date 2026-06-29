use cuda_device::{SharedArray, grid};

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS};

use super::super::super::threads::WARPS_PER_BLOCK;
use super::super::super::work_grid::WorkGrid;
use super::momentum::momentum_orient;
use super::polar_step::run_polar_step;
use super::quant::quantize_updated_master;
use super::update::update_master_chunks;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn aurora_matrix_update_body(
    grad: *const f32,
    momentum: *mut f32,
    z_master: *mut f32,
    x_master: *mut f32,
    out_fp4: *mut u8,
    out_scales: *mut u8,
    out_global_scale: *mut f32,
    oriented: *mut f32,
    polar_next: *mut f32,
    polar_x: *mut f32,
    polar_gram: *mut f32,
    polar_ax: *mut f32,
    polar_chunks: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
    rows: u32,
    cols: u32,
    mu: f32,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    iterations: u32,
) {
    let len = rows * cols;
    let transposed = rows < cols;
    let oriented_ptr = oriented;

    momentum_orient(
        grad,
        momentum,
        oriented_ptr,
        work,
        rows,
        cols,
        mu,
        transposed,
    );
    grid::sync();

    let polar_update = run_polar_step(
        oriented_ptr,
        polar_next,
        polar_x,
        polar_gram,
        polar_ax,
        polar_chunks,
        a_tile,
        b_tile,
        warp_sums,
        work,
        rows,
        cols,
        transposed,
        iterations,
    );

    // Polar Express returns after a grid-wide sync; update can consume its result directly.
    update_master_chunks(
        polar_update,
        z_master,
        x_master,
        polar_chunks,
        rows,
        cols,
        len,
        rows > cols,
        learning_rate,
        weight_decay,
        average_coefficient,
        warp_sums,
        work,
    );
    grid::sync();

    quantize_updated_master(
        x_master,
        polar_chunks,
        out_fp4,
        out_scales,
        out_global_scale,
        len,
        warp_sums,
        work,
    );
}
