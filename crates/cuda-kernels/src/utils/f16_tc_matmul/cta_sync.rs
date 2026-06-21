use cuda_device::thread;

use super::cta_tile::CTA_K;

#[inline(always)]
pub(super) fn sync_before_next_k(k_base: u32, k: u32) {
    if k_base + CTA_K < k {
        thread::sync_threads();
    }
}
