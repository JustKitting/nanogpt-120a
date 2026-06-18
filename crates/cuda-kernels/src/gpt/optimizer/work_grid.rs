use cuda_device::thread;

use crate::f16_tc_matmul::cta_tile::CTA_THREADS;

#[derive(Clone, Copy)]
pub(super) struct WorkGrid {
    block: u32,
    blocks: u32,
}

impl WorkGrid {
    #[inline(always)]
    pub(super) fn x_axis() -> Self {
        Self {
            block: thread::blockIdx_x(),
            blocks: thread::gridDim_x(),
        }
    }

    #[inline(always)]
    pub(super) fn block(self) -> u32 {
        self.block
    }

    #[inline(always)]
    pub(super) fn blocks(self) -> u32 {
        self.blocks
    }

    #[inline(always)]
    pub(super) fn thread(self) -> u32 {
        self.block * CTA_THREADS + thread::threadIdx_x()
    }

    #[inline(always)]
    pub(super) fn stride(self) -> u32 {
        self.blocks * CTA_THREADS
    }
}
