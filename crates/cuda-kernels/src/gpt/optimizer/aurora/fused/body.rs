use cuda_device::{SharedArray, grid};

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS};

use super::super::super::threads::WARPS_PER_BLOCK;
use super::super::super::work_grid::WorkGrid;
use super::momentum::momentum_orient;
use super::polar_step::run_polar_step;
use super::quant::quantize_updated_master;
use super::update::update_master_chunks;

#[derive(Clone, Copy)]
pub(super) struct AuroraMatrixState {
    pub grad: *const f32, pub momentum: *mut f32, pub z_master: *mut f32, pub x_master: *mut f32,
    pub out_fp4: *mut u8, pub out_scales: *mut u8, pub out_global_scale: *mut f32,
}

#[derive(Clone, Copy)]
pub(super) struct AuroraMatrixScratch {
    pub oriented: *mut f32, pub polar_next: *mut f32, pub polar_x: *mut f32,
    pub polar_gram: *mut f32, pub polar_ax: *mut f32, pub polar_chunks: *mut f32,
}

pub(super) struct AuroraMatrixTiles<'a> {
    pub a_tile: &'a mut SharedArray<u16, CTA_A_ELEMS>,
    pub b_tile: &'a mut SharedArray<u16, CTA_B_ELEMS>,
    pub warp_sums: &'a mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
}

#[derive(Clone, Copy)]
pub(super) struct AuroraMatrixShape { pub rows: u32, pub cols: u32 }

#[derive(Clone, Copy)]
pub(super) struct AuroraUpdateScalars {
    pub mu: f32, pub learning_rate: f32, pub weight_decay: f32,
    pub average_coefficient: f32, pub iterations: u32,
}

pub(super) fn aurora_matrix_update_body(
    state: AuroraMatrixState,
    scratch: AuroraMatrixScratch,
    tiles: AuroraMatrixTiles<'_>,
    work: WorkGrid,
    shape: AuroraMatrixShape,
    scalars: AuroraUpdateScalars,
) {
    let len = shape.rows * shape.cols;
    let transposed = shape.rows < shape.cols;

    momentum_orient(
        state.grad,
        state.momentum,
        scratch.oriented,
        work,
        shape,
        scalars.mu,
        transposed,
    );
    grid::sync();

    let polar_update = run_polar_step(
        scratch.oriented,
        scratch.polar_next,
        scratch.polar_x,
        scratch.polar_gram,
        scratch.polar_ax,
        scratch.polar_chunks,
        tiles.a_tile,
        tiles.b_tile,
        tiles.warp_sums,
        work,
        shape.rows,
        shape.cols,
        transposed,
        scalars.iterations,
    );

    // Polar Express returns after a grid-wide sync; update can consume its result directly.
    update_master_chunks(
        polar_update,
        state.z_master,
        state.x_master,
        scratch.polar_chunks,
        shape.rows,
        shape.cols,
        len,
        shape.rows > shape.cols,
        scalars.learning_rate,
        scalars.weight_decay,
        scalars.average_coefficient,
        tiles.warp_sums,
        work,
    );
    grid::sync();

    quantize_updated_master(
        state.x_master,
        scratch.polar_chunks,
        state.out_fp4,
        state.out_scales,
        state.out_global_scale,
        len,
        tiles.warp_sums,
        work,
    );
}
