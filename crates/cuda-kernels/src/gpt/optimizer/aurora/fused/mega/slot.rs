use crate::optimizer::AuroraSlotDescriptor;

use super::super::super::super::work_grid::WorkGrid;
use super::super::body::aurora_matrix_update_body;
use super::super::types::{AuroraMatrixScratch, AuroraMatrixShape, AuroraMatrixState, AuroraMatrixTiles, AuroraUpdateScalars};

#[derive(Clone, Copy)]
pub(super) struct MegaScratchLayout { pub max_len: u32, pub max_ax_len: u32, pub max_dim: u32 }

pub(super) fn launch_slot(
    slot: u32,
    scratch_slot: u32,
    slots: &[AuroraSlotDescriptor],
    scratch: AuroraMatrixScratch,
    layout: MegaScratchLayout,
    tiles: AuroraMatrixTiles<'_>,
    scalars: AuroraUpdateScalars,
) {
    let desc = slots[slot as usize];
    let scalars = AuroraUpdateScalars {
        learning_rate: scalars.learning_rate * desc.learning_rate_multiplier,
        ..scalars
    };
    let work = WorkGrid::x_axis();
    aurora_matrix_update_body(
        AuroraMatrixState {
            grad: ptr_const(desc.grad), momentum: ptr_mut(desc.momentum),
            z_master: ptr_mut(desc.z_master), x_master: ptr_mut(desc.x_master),
            out_fp4: ptr_mut(desc.bytes), out_scales: ptr_mut(desc.scales),
            out_global_scale: ptr_mut(desc.global_scale),
        },
        AuroraMatrixScratch {
            oriented: offset_ptr(scratch.oriented, scratch_slot, layout.max_len as usize),
            polar_next: offset_ptr(scratch.polar_next, scratch_slot, layout.max_len as usize),
            polar_x: offset_ptr(scratch.polar_x, scratch_slot, layout.max_len as usize),
            polar_gram: offset_ptr(scratch.polar_gram, scratch_slot, (layout.max_dim * layout.max_dim) as usize),
            polar_ax: offset_ptr(scratch.polar_ax, scratch_slot, layout.max_ax_len as usize),
            polar_chunks: offset_ptr(scratch.polar_chunks, scratch_slot, work.blocks() as usize),
        },
        tiles,
        work,
        AuroraMatrixShape { rows: desc.rows, cols: desc.cols },
        scalars,
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
