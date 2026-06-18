use cuda_device::thread;

pub(super) const CTA_M: u32 = 64;
pub(super) const CTA_N: u32 = 64;
pub(super) const CTA_K: u32 = 16;
pub(super) const CTA_THREADS: u32 = 256;
pub(super) const CTA_A_ELEMS: usize = CTA_M as usize * CTA_K as usize;
pub(super) const CTA_B_ELEMS: usize = CTA_N as usize * CTA_K as usize;

#[derive(Clone, Copy)]
pub(super) struct CtaTile {
    pub batch: u32,
    pub row_base: u32,
    pub col_base: u32,
    pub warp_m: u32,
    pub warp_n0: u32,
    pub group: u32,
    pub thread_in_group: u32,
}

impl CtaTile {
    pub(super) fn new(thread_id: u32) -> Self {
        let lane = thread_id & 31;
        let warp = thread_id >> 5;
        Self {
            batch: thread::blockIdx_z(),
            row_base: thread::blockIdx_y() * CTA_M,
            col_base: thread::blockIdx_x() * CTA_N,
            warp_m: warp >> 1,
            warp_n0: (warp & 1) << 2,
            group: lane >> 2,
            thread_in_group: lane & 0x3,
        }
    }
}
