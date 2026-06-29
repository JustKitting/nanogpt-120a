use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::layer_norm_reduce::layer_norm_block_reduce;
use crate::layer_norm_utils::{
    f16_column, f32_column, layer_norm_columns3, layer_norm_map3, layer_norm_map3_indexed,
    layer_norm_store3, layer_norm_sum3, nvfp4_column,
};
use crate::warp_reduce::warp_sum_f32;

pub const THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const WARPS_PER_BLOCK: u32 = THREADS_PER_BLOCK / WARP_SIZE;

#[allow(static_mut_refs)]
#[cuda_module]
pub(super) mod kernels {
    use super::*;

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn layer_norm_backward_input_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut d_residual: DisjointSlice<f32>,
        row_count: u32,
        embedding_dim: u32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / WARP_SIZE;

        if row < row_count {
            let row_base = row as usize * embedding_dim as usize;
            let cols = layer_norm_columns3!(thread, THREADS_PER_BLOCK);
            let row_mean = mean[row as usize];
            let row_inv_std = inv_std[row as usize];
            let xhat = layer_norm_map3!(cols, |col| {
                (f16_column(residual, row_base, col, embedding_dim) - row_mean) * row_inv_std
            });
            let dxhat = layer_norm_map3!(cols, |col| {
                let grad = f32_column(d_normalized, row_base, col, embedding_dim);
                let weight = nvfp4_column(
                    weight_bytes,
                    weight_scales,
                    weight_global_scale[0],
                    0,
                    col,
                    embedding_dim,
                );
                grad * weight
            });
            let dxhat_sum = layer_norm_block_reduce!(
                WARP_SUMS,
                WARPS_PER_BLOCK,
                layer_norm_sum3!(dxhat),
                lane,
                warp_in_block,
                warp_sum_f32
            );
            let xhat_dxhat = layer_norm_map3_indexed!(xhat, |index, value| value * dxhat[index]);
            let xhat_dxhat_sum = layer_norm_block_reduce!(
                WARP_SUMS,
                WARPS_PER_BLOCK,
                layer_norm_sum3!(xhat_dxhat),
                lane,
                warp_in_block,
                warp_sum_f32
            );
            let inv_dim = 1.0 / embedding_dim as f32;
            let dx = layer_norm_map3_indexed!(dxhat, |index, value| {
                (value - dxhat_sum * inv_dim - xhat[index] * xhat_dxhat_sum * inv_dim) * row_inv_std
            });

            layer_norm_store3!(&mut d_residual, row_base, cols, embedding_dim, dx);
        }
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn layer_norm_backward_input_f32_kernel(
        residual: &[f32],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut d_residual: DisjointSlice<f32>,
        row_count: u32,
        embedding_dim: u32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / WARP_SIZE;

        if row < row_count {
            let row_base = row as usize * embedding_dim as usize;
            let cols = layer_norm_columns3!(thread, THREADS_PER_BLOCK);
            let row_mean = mean[row as usize];
            let row_inv_std = inv_std[row as usize];
            let xhat = layer_norm_map3!(cols, |col| {
                (f32_column(residual, row_base, col, embedding_dim) - row_mean) * row_inv_std
            });
            let dxhat = layer_norm_map3!(cols, |col| {
                let grad = f32_column(d_normalized, row_base, col, embedding_dim);
                let weight = nvfp4_column(
                    weight_bytes,
                    weight_scales,
                    weight_global_scale[0],
                    0,
                    col,
                    embedding_dim,
                );
                grad * weight
            });
            let dxhat_sum = layer_norm_block_reduce!(
                WARP_SUMS,
                WARPS_PER_BLOCK,
                layer_norm_sum3!(dxhat),
                lane,
                warp_in_block,
                warp_sum_f32
            );
            let xhat_dxhat = layer_norm_map3_indexed!(xhat, |index, value| value * dxhat[index]);
            let xhat_dxhat_sum = layer_norm_block_reduce!(
                WARP_SUMS,
                WARPS_PER_BLOCK,
                layer_norm_sum3!(xhat_dxhat),
                lane,
                warp_in_block,
                warp_sum_f32
            );
            let inv_dim = 1.0 / embedding_dim as f32;
            let dx = layer_norm_map3_indexed!(dxhat, |index, value| {
                (value - dxhat_sum * inv_dim - xhat[index] * xhat_dxhat_sum * inv_dim) * row_inv_std
            });

            layer_norm_store3!(&mut d_residual, row_base, cols, embedding_dim, dx);
        }
    }
}
