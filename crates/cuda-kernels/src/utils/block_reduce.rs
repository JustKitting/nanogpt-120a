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

macro_rules! block_sum_f32 {
    ($storage:ident, $local:expr, $lane:expr, $warp:expr, $warps:expr) => {
        crate::block_reduce::block_reduce_f32!(
            $storage,
            $warps,
            $local,
            $lane,
            $warp,
            crate::warp_reduce::warp_sum_f32,
            0.0
        )
    };
}

pub(crate) use block_sum_f32;

macro_rules! block_max_f32 {
    ($storage:ident, $local:expr, $lane:expr, $warp:expr, $warps:expr) => {
        crate::block_reduce::block_reduce_f32!(
            $storage,
            $warps,
            $local,
            $lane,
            $warp,
            crate::warp_reduce::warp_max_f32,
            0.0
        )
    };
}

pub(crate) use block_max_f32;
