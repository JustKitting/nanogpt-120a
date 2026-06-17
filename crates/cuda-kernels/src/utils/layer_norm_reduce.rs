macro_rules! layer_norm_block_reduce {
    ($storage:ident, $warps:expr, $local:expr, $lane:expr, $warp:expr, $op:ident) => {{
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
            0.0
        };
        let block_total = $op(partial);
        if $warp == 0 && $lane == 0 {
            unsafe {
                $storage[0] = block_total;
            }
        }
        cuda_device::thread::sync_threads();
        unsafe { $storage[0] }
    }};
}

pub(crate) use layer_norm_block_reduce;

macro_rules! layer_norm_store_row {
    ($out:expr, $row:expr, $lane:expr, $warp:expr, $value:expr) => {
        if $warp == 0 && $lane == 0 {
            unsafe {
                *$out.get_unchecked_mut($row as usize) = $value;
            }
        }
    };
}

pub(crate) use layer_norm_store_row;
