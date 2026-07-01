use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::block_reduce::block_sum_shared_f32;
use crate::device_ptr::{read_f32, write_f32};
use crate::float_ptx::sqrt_f32;
use crate::warp_reduce::thread_lane_warp;

const THREADS_PER_BLOCK: u32 = 256;
const WARPS_PER_BLOCK: u32 = 8;
const WARP_SUM_SLOTS: usize = WARPS_PER_BLOCK as usize;
const VALUES_PER_CHUNK: u32 = 1024;
const APPLY_UNROLL: u32 = 4;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn grad_clip_sumsq_chunks_kernel(
        ptrs: &[u64], lens: &[u32], chunk_offsets: &[u32],
        mut chunk_sums: DisjointSlice<f32>,
        slot_count: u32, chunk_count: u32,
    ) {
        let chunk = thread::blockIdx_x();
        if chunk >= chunk_count {
            return;
        }

        let slot = slot_for_chunk(chunk_offsets, slot_count, chunk);
        let local_chunk = chunk - chunk_offsets[slot as usize];
        let ptr = ptrs[slot as usize] as *const f32;
        let len = lens[slot as usize];
        let base = local_chunk * VALUES_PER_CHUNK;
        let (thread, lane, warp) = thread_lane_warp();
        static mut WARP_SUMS: SharedArray<f32, WARP_SUM_SLOTS> = SharedArray::UNINIT;
        let mut local = 0.0;
        let mut offset = thread;

        while offset < VALUES_PER_CHUNK {
            let index = base + offset;
            if index < len {
                let value = read_f32(ptr, index);
                local += value * value;
            }
            offset += THREADS_PER_BLOCK;
        }

        let sum = unsafe { block_sum_shared_f32(&mut WARP_SUMS, local, lane, warp) };
        if thread::threadIdx_x() == 0 {
            write_f32(chunk_sums.as_mut_ptr(), chunk, sum);
        }
    }

    #[kernel]
    pub fn grad_clip_scale_kernel(
        chunk_sums: &[f32], mut scale: DisjointSlice<f32>, mut norm_out: DisjointSlice<f32>,
        chunk_count: u32, max_norm: f32,
    ) {
        let (thread, lane, warp) = thread_lane_warp();
        static mut WARP_SUMS: SharedArray<f32, WARP_SUM_SLOTS> = SharedArray::UNINIT;
        let mut local = 0.0;
        let mut chunk = thread;

        while chunk < chunk_count {
            local += chunk_sums[chunk as usize];
            chunk += THREADS_PER_BLOCK;
        }

        let sum = unsafe { block_sum_shared_f32(&mut WARP_SUMS, local, lane, warp) };
        if thread::threadIdx_x() == 0 {
            let norm = sqrt_f32(sum);
            let value = if norm > max_norm {
                max_norm / (norm + 1.0e-6)
            } else {
                1.0
            };
            write_f32(scale.as_mut_ptr(), 0, value);
            write_f32(norm_out.as_mut_ptr(), 0, norm);
        }
    }

    #[kernel]
    pub fn grad_clip_apply_kernel(
        ptrs: &[u64], lens: &[u32], chunk_offsets: &[u32], scale: &[f32],
        slot_count: u32, chunk_count: u32,
    ) {
        let chunk = thread::blockIdx_x();
        if chunk >= chunk_count {
            return;
        }

        let slot = slot_for_chunk(chunk_offsets, slot_count, chunk);
        let local_chunk = chunk - chunk_offsets[slot as usize];
        let ptr = ptrs[slot as usize] as *mut f32;
        let len = lens[slot as usize];
        let base = local_chunk * VALUES_PER_CHUNK;
        let multiplier = scale[0];
        let mut offset = thread::threadIdx_x();

        while offset < VALUES_PER_CHUNK {
            apply_one(ptr, len, base + offset, multiplier);
            apply_one(ptr, len, base + offset + THREADS_PER_BLOCK, multiplier);
            apply_one(ptr, len, base + offset + THREADS_PER_BLOCK * 2, multiplier);
            apply_one(ptr, len, base + offset + THREADS_PER_BLOCK * 3, multiplier);
            offset += THREADS_PER_BLOCK * APPLY_UNROLL;
        }
    }

    #[inline(always)]
    fn apply_one(ptr: *mut f32, len: u32, index: u32, multiplier: f32) {
        if index < len {
            let value = read_f32(ptr as *const f32, index) * multiplier;
            write_f32(ptr, index, value);
        }
    }

    #[inline(always)]
    fn slot_for_chunk(chunk_offsets: &[u32], slot_count: u32, chunk: u32) -> u32 {
        let mut slot = 0;
        while slot + 1 < slot_count && chunk_offsets[(slot + 1) as usize] <= chunk {
            slot += 1;
        }
        slot
    }
}

pub const GRAD_CLIP_VALUES_PER_CHUNK: usize = VALUES_PER_CHUNK as usize;
pub(super) const GRAD_CLIP_THREADS_PER_BLOCK: u32 = THREADS_PER_BLOCK;
