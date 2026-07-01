use cuda_device::thread;

use crate::device_ptr::{read_f32, write_f32};

use super::layout::slot_for_chunk;
use super::{APPLY_UNROLL, THREADS_PER_BLOCK, VALUES_PER_CHUNK};

pub(super) fn grad_clip_apply_body(
    ptrs: &[u64],
    lens: &[u32],
    chunk_offsets: &[u32],
    scale: &[f32],
    slot_count: u32,
    chunk_count: u32,
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
