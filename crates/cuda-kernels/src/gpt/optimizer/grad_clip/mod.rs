mod apply;
mod layout;
mod scale;
mod sumsq;

use cuda_device::cuda_module;

pub(super) const THREADS_PER_BLOCK: u32 = 256;
const WARPS_PER_BLOCK: u32 = 8;
pub(super) const WARP_SUM_SLOTS: usize = WARPS_PER_BLOCK as usize;
pub(super) const VALUES_PER_CHUNK: u32 = 1024;
pub(super) const APPLY_UNROLL: u32 = 4;

#[cuda_module]
pub(super) mod module {
    use cuda_device::{DisjointSlice, kernel};

    use super::apply::grad_clip_apply_body;
    use super::scale::grad_clip_scale_body;
    use super::sumsq::grad_clip_sumsq_chunks_body;

    #[kernel]
    pub fn grad_clip_sumsq_chunks_kernel(
        ptrs: &[u64],
        lens: &[u32],
        chunk_offsets: &[u32],
        mut chunk_sums: DisjointSlice<f32>,
        slot_count: u32,
        chunk_count: u32,
    ) {
        grad_clip_sumsq_chunks_body(
            ptrs,
            lens,
            chunk_offsets,
            &mut chunk_sums,
            slot_count,
            chunk_count,
        );
    }

    #[kernel]
    pub fn grad_clip_scale_kernel(
        chunk_sums: &[f32],
        mut scale: DisjointSlice<f32>,
        mut norm_out: DisjointSlice<f32>,
        chunk_count: u32,
        max_norm: f32,
    ) {
        grad_clip_scale_body(chunk_sums, &mut scale, &mut norm_out, chunk_count, max_norm);
    }

    #[kernel]
    pub fn grad_clip_apply_kernel(
        ptrs: &[u64],
        lens: &[u32],
        chunk_offsets: &[u32],
        scale: &[f32],
        slot_count: u32,
        chunk_count: u32,
    ) {
        grad_clip_apply_body(ptrs, lens, chunk_offsets, scale, slot_count, chunk_count);
    }
}

pub const GRAD_CLIP_VALUES_PER_CHUNK: usize = VALUES_PER_CHUNK as usize;
pub(super) const GRAD_CLIP_THREADS_PER_BLOCK: u32 = THREADS_PER_BLOCK;
