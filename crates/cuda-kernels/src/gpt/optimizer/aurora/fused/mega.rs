use cuda_device::{
    DisjointSlice, SharedArray, cooperative_launch, cuda_module, grid, kernel, thread,
};

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS};
use crate::optimizer::AuroraSlotDescriptor;

use super::super::super::AURORA_MATRIX_PHASES;
use super::super::super::threads::WARPS_PER_BLOCK;

mod slot;

#[allow(static_mut_refs)]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    #[cooperative_launch]
    pub fn aurora_mega_update_cooperative_kernel(
        slots: &[AuroraSlotDescriptor],
        mut oriented: DisjointSlice<f32>,
        mut polar_next: DisjointSlice<f32>,
        mut polar_x: DisjointSlice<f32>,
        mut polar_gram: DisjointSlice<f32>,
        mut polar_ax: DisjointSlice<f32>,
        mut polar_chunks: DisjointSlice<f32>,
        slot_count: u32,
        max_len: u32,
        max_ax_len: u32,
        max_dim: u32,
        mu: f32,
        learning_rate: f32,
        weight_decay: f32,
        average_coefficient: f32,
        iterations: u32,
    ) {
        static mut A_TILE: SharedArray<u16, CTA_A_ELEMS> = SharedArray::UNINIT;
        static mut B_TILE: SharedArray<u16, CTA_B_ELEMS> = SharedArray::UNINIT;
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let matrix = thread::blockIdx_y();
        let matrix_count = thread::gridDim_y();
        let mut phase = 0;
        while phase < AURORA_MATRIX_PHASES as u32 {
            let slot = phase * matrix_count + matrix;
            if slot < slot_count {
                unsafe {
                    slot::launch_slot(
                        slot,
                        matrix,
                        slots,
                        oriented.as_mut_ptr(),
                        polar_next.as_mut_ptr(),
                        polar_x.as_mut_ptr(),
                        polar_gram.as_mut_ptr(),
                        polar_ax.as_mut_ptr(),
                        polar_chunks.as_mut_ptr(),
                        &mut A_TILE,
                        &mut B_TILE,
                        &mut WARP_SUMS,
                        max_len,
                        max_ax_len,
                        max_dim,
                        mu,
                        learning_rate,
                        weight_decay,
                        average_coefficient,
                        iterations,
                    );
                }
            }
            grid::sync();
            phase += 1;
        }
    }
}
