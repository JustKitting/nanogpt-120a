use cuda_device::SharedArray;

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS};
use crate::optimizer::AuroraSlotDescriptor;

use super::super::super::super::threads::WARPS_PER_BLOCK;
use super::super::super::super::work_grid::WorkGrid;
use super::super::body::aurora_matrix_update_body;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn launch_slot(
    slot: u32,
    scratch_slot: u32,
    slots: &[AuroraSlotDescriptor],
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
    let desc = slots[slot as usize];
    let learning_rate = learning_rate * desc.learning_rate_multiplier;
    let work = WorkGrid::x_axis();
    aurora_matrix_update_body(
        ptr_const(desc.grad),
        ptr_mut(desc.momentum),
        ptr_mut(desc.z_master),
        ptr_mut(desc.x_master),
        ptr_mut(desc.bytes),
        ptr_mut(desc.scales),
        ptr_mut(desc.global_scale),
        offset_ptr(oriented, scratch_slot, max_len as usize),
        offset_ptr(polar_next, scratch_slot, max_len as usize),
        offset_ptr(polar_x, scratch_slot, max_len as usize),
        offset_ptr(polar_gram, scratch_slot, (max_dim * max_dim) as usize),
        offset_ptr(polar_ax, scratch_slot, max_ax_len as usize),
        offset_ptr(polar_chunks, scratch_slot, work.blocks() as usize),
        a_tile,
        b_tile,
        warp_sums,
        work,
        desc.rows,
        desc.cols,
        mu,
        learning_rate,
        weight_decay,
        average_coefficient,
        iterations,
    );
}

#[inline(always)]
fn ptr_const<T>(ptr: u64) -> *const T {
    ptr as usize as *const T
}

#[inline(always)]
fn ptr_mut<T>(ptr: u64) -> *mut T {
    ptr as usize as *mut T
}

#[inline(always)]
fn offset_ptr<T>(ptr: *mut T, slot: u32, len: usize) -> *mut T {
    unsafe { ptr.add(slot as usize * len) }
}
