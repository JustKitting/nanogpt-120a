macro_rules! dispatch_fp32_pair {
    (
        row_grid_dim: $row_grid_dim:expr,
        x: $x:expr,
        output: [$out_fp4:expr, $out_scales:expr, $out_global_scales:expr],
        transpose_output: [$transpose_out_fp4:expr, $transpose_out_scales:expr, $transpose_out_global_scales:expr],
        scale: [
            $global_scale:expr, $scale_override:expr, $sign_seed:expr, $scale_seed:expr, $transpose_scale_seed:expr
        ],
        row: $row_body:ident($($row_arg:expr),* $(,)?);
        transpose: $transpose_body:ident($($transpose_arg:expr),* $(,)?)
    ) => {{
        let block = cuda_device::thread::blockIdx_x();
        let warp_in_block = cuda_device::thread::threadIdx_x() / 32;
        if block < $row_grid_dim {
            let chunk = block * super::AMAX_WARPS_PER_BLOCK + warp_in_block;
            $row_body(
                $x,
                &mut $out_fp4,
                &mut $out_scales,
                &mut $out_global_scales,
                chunk,
                $($row_arg,)*
                $global_scale[0],
                $scale_override,
                $sign_seed,
                $scale_seed,
            );
        } else {
            let chunk = (block - $row_grid_dim) * super::AMAX_WARPS_PER_BLOCK + warp_in_block;
            $transpose_body(
                $x,
                &mut $transpose_out_fp4,
                &mut $transpose_out_scales,
                &mut $transpose_out_global_scales,
                chunk,
                $($transpose_arg,)*
                $global_scale[0],
                $scale_override,
                $sign_seed,
                $transpose_scale_seed,
            );
        }
    }};
}

pub(super) use dispatch_fp32_pair;
