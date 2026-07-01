#[inline(always)]
pub(super) fn slot_for_chunk(chunk_offsets: &[u32], slot_count: u32, chunk: u32) -> u32 {
    let mut slot = 0;
    while slot + 1 < slot_count && chunk_offsets[(slot + 1) as usize] <= chunk {
        slot += 1;
    }
    slot
}
