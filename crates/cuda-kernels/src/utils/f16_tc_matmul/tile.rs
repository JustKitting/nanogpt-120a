use cuda_device::thread;

pub(super) const M: u32 = 16;
pub(super) const N: u32 = 8;
pub(super) const K_STEP: u32 = 16;

#[derive(Clone, Copy)]
pub(super) struct Tile {
    pub row: u32,
    pub col: u32,
    pub group: u32,
    pub thread_in_group: u32,
}

impl Tile {
    pub(super) fn new(lane: u32) -> Self {
        Self {
            row: thread::blockIdx_y() * M,
            col: thread::blockIdx_x() * N,
            group: lane >> 2,
            thread_in_group: lane & 0x3,
        }
    }
}
