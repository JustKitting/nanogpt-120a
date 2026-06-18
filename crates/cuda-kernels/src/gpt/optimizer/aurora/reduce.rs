macro_rules! block_sum {
    ($storage:ident, $local:expr, $lane:expr, $warp:expr) => {{
        let warp_total = crate::warp_reduce::warp_sum_f32($local);
        if $lane == 0 {
            unsafe {
                $storage[$warp as usize] = warp_total;
            }
        }
        thread::sync_threads();

        let partial = if $warp == 0 && $lane < WARPS_PER_BLOCK {
            unsafe { $storage[$lane as usize] }
        } else {
            0.0
        };
        let block_total = crate::warp_reduce::warp_sum_f32(partial);
        thread::sync_threads();
        block_total
    }};
}
