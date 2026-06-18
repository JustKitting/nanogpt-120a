use cuda_device::SharedArray;

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS};

use super::super::super::super::threads::WARPS_PER_BLOCK;
use super::super::super::super::work_grid::WorkGrid;
use super::super::body::aurora_matrix_update_body;

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_slot(
    slot: u32,
    scratch_slot: u32,
    grad_ptrs: &[u64],
    momentum_ptrs: &[u64],
    z_master_ptrs: &[u64],
    x_master_ptrs: &[u64],
    byte_ptrs: &[u64],
    scale_ptrs: &[u64],
    global_scale_ptrs: &[u64],
    rows: &[u32],
    cols: &[u32],
    oriented: *mut f32,
    polar_next: *mut f32,
    polar_x: *mut f32,
    polar_gram: *mut f32,
    polar_ax: *mut f32,
    polar_chunks: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    max_len: u32,
    max_ax_len: u32,
    max_dim: u32,
    mu: f32,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    iterations: u32,
) {
    let rows = rows[slot as usize];
    let cols = cols[slot as usize];
    aurora_matrix_update_body(
        ptr_const(grad_ptrs, slot),
        ptr_mut(momentum_ptrs, slot),
        ptr_mut(z_master_ptrs, slot),
        ptr_mut(x_master_ptrs, slot),
        ptr_mut(byte_ptrs, slot),
        ptr_mut(scale_ptrs, slot),
        ptr_mut(global_scale_ptrs, slot),
        offset_ptr(oriented, scratch_slot, max_len as usize),
        offset_ptr(polar_next, scratch_slot, max_len as usize),
        offset_ptr(polar_x, scratch_slot, max_len as usize),
        offset_ptr(polar_gram, scratch_slot, (max_dim * max_dim) as usize),
        offset_ptr(polar_ax, scratch_slot, max_ax_len as usize),
        offset_ptr(
            polar_chunks,
            scratch_slot,
            WorkGrid::x_axis().blocks() as usize,
        ),
        a_tile,
        b_tile,
        warp_sums,
        WorkGrid::x_axis(),
        rows,
        cols,
        mu,
        learning_rate,
        weight_decay,
        average_coefficient,
        iterations,
    );
}

#[inline(always)]
fn ptr_const<T>(ptrs: &[u64], slot: u32) -> *const T {
    ptrs[slot as usize] as usize as *const T
}

#[inline(always)]
fn ptr_mut<T>(ptrs: &[u64], slot: u32) -> *mut T {
    ptrs[slot as usize] as usize as *mut T
}

#[inline(always)]
fn offset_ptr<T>(ptr: *mut T, slot: u32, len: usize) -> *mut T {
    unsafe { ptr.add(slot as usize * len) }
}
