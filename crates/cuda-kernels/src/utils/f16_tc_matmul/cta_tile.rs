use cuda_device::thread;

pub(crate) const CTA_M: u32 = 64;
pub(crate) const CTA_N: u32 = 64;
pub(crate) const CTA_K: u32 = 16;
pub(crate) const CTA_THREADS: u32 = 256;
pub(crate) const CTA_A_ELEMS: usize = CTA_M as usize * CTA_K as usize;
pub(crate) const CTA_B_ELEMS: usize = CTA_N as usize * CTA_K as usize;

#[derive(Clone, Copy)]
pub(crate) struct CtaTile {
    pub(crate) batch: u32,
    pub(crate) row_base: u32,
    pub(crate) col_base: u32,
    pub(crate) warp_m: u32,
    pub(crate) warp_n0: u32,
    pub(crate) group: u32,
    pub(crate) thread_in_group: u32,
}

impl CtaTile {
    pub(super) fn new(thread_id: u32) -> Self {
        Self::from_tile(
            thread_id,
            thread::blockIdx_y(),
            thread::blockIdx_x(),
            thread::blockIdx_z(),
        )
    }

    pub(crate) fn from_tile(thread_id: u32, tile_row: u32, tile_col: u32, batch: u32) -> Self {
        let lane = thread_id & 31;
        let warp = thread_id >> 5;
        Self {
            batch,
            row_base: tile_row * CTA_M,
            col_base: tile_col * CTA_N,
            warp_m: warp >> 1,
            warp_n0: (warp & 1) << 2,
            group: lane >> 2,
            thread_in_group: lane & 0x3,
        }
    }
}
