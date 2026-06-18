macro_rules! layer_norm_block_reduce {
    ($storage:ident, $warps:expr, $local:expr, $lane:expr, $warp:expr, $op:path) => {
        crate::block_reduce::block_reduce_f32!($storage, $warps, $local, $lane, $warp, $op, 0.0)
    };
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
