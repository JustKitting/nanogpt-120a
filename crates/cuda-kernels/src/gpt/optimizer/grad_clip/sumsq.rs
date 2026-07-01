use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::block_reduce::block_sum_shared_f32;
use crate::device_ptr::{read_f32, write_f32};
use crate::warp_reduce::thread_lane_warp;

use super::layout::slot_for_chunk;
use super::{THREADS_PER_BLOCK, VALUES_PER_CHUNK, WARP_SUM_SLOTS};

pub(super) fn grad_clip_sumsq_chunks_body(
    ptrs: &[u64],
    lens: &[u32],
    chunk_offsets: &[u32],
    chunk_sums: &mut DisjointSlice<f32>,
    slot_count: u32,
    chunk_count: u32,
) {
    static mut WARP_SUMS: SharedArray<f32, WARP_SUM_SLOTS> = SharedArray::UNINIT;

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
