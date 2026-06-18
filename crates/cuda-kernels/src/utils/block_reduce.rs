macro_rules! block_reduce_f32 {
    ($storage:ident, $warps:expr, $local:expr, $lane:expr, $warp:expr, $op:path, $identity:expr) => {{
        let warp_total = $op($local);
        if $lane == 0 {
            unsafe {
                $storage[$warp as usize] = warp_total;
            }
        }
        cuda_device::thread::sync_threads();

        let partial = if $warp == 0 && $lane < $warps {
            unsafe { $storage[$lane as usize] }
        } else {
            $identity
        };
        let block_total = $op(partial);
        if $warp == 0 && $lane == 0 {
            unsafe {
                $storage[0] = block_total;
            }
        }
        cuda_device::thread::sync_threads();

        let block_total = unsafe { $storage[0] };
        cuda_device::thread::sync_threads();
        block_total
    }};
}

pub(crate) use block_reduce_f32;

use cuda_device::{SharedArray, thread};

use crate::warp_reduce::{warp_max_f32, warp_sum_f32};

#[inline(always)]
pub(crate) fn block_sum_shared_f32<const WARPS: usize>(
    storage: &mut SharedArray<f32, WARPS>,
    local: f32,
    lane: u32,
    warp: u32,
) -> f32 {
    let warp_total = warp_sum_f32(local);
    if lane == 0 {
        storage[warp as usize] = warp_total;
    }
    thread::sync_threads();

    let partial = if warp == 0 && lane < WARPS as u32 {
        storage[lane as usize]
    } else {
        0.0
    };
    let block_total = warp_sum_f32(partial);
    if warp == 0 && lane == 0 {
        storage[0] = block_total;
    }
    thread::sync_threads();

    let block_total = storage[0];
    thread::sync_threads();
    block_total
}

#[inline(always)]
pub(crate) fn block_max_shared_f32<const WARPS: usize>(
    storage: &mut SharedArray<f32, WARPS>,
    local: f32,
    lane: u32,
    warp: u32,
) -> f32 {
    let warp_total = warp_max_f32(local);
    if lane == 0 {
        storage[warp as usize] = warp_total;
    }
    thread::sync_threads();

    let partial = if warp == 0 && lane < WARPS as u32 {
        storage[lane as usize]
    } else {
        0.0
    };
    let block_total = warp_max_f32(partial);
    if warp == 0 && lane == 0 {
        storage[0] = block_total;
    }
    thread::sync_threads();

    let block_total = storage[0];
    thread::sync_threads();
    block_total
}
