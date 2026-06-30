use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::float_ptx::{fma_f32, sqrt_f32};
use crate::layer_norm_reduce::{layer_norm_block_reduce, layer_norm_store_row};
use crate::layer_norm_utils::{
    centered_column, f32_column, layer_norm_columns3, layer_norm_map3, layer_norm_map3_indexed,
    layer_norm_square_sum3, layer_norm_store_f16_3, layer_norm_store3, layer_norm_sum3, max_abs3,
    nvfp4_affine_normalized_column,
};
use crate::warp_reduce::{warp_max_f32, warp_sum_f32};

pub const ROW_SIZE: usize = 32;
const WARPS_PER_BLOCK: u32 = 8;
const THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * ROW_SIZE as u32;
const GPT_LAYER_NORM_THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const GPT_LAYER_NORM_WARPS_PER_BLOCK: u32 = GPT_LAYER_NORM_THREADS_PER_BLOCK / WARP_SIZE;

#[path = "layer_norm/launcher.rs"]
mod launcher;
pub use launcher::{
    GptLayerNormArgs, GptLayerNormSaveResidualF16Args, LayerNormArgs, LayerNormModule,
};

#[allow(static_mut_refs)]
#[cuda_module]
mod kernels {
    use super::*;

    const ROW_SIZE_F32: f32 = ROW_SIZE as f32;

    #[kernel]
    pub fn layer_norm_warp_f32_kernel(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut out: DisjointSlice<f32>,
        row_count: u32,
        epsilon: f32,
    ) {
        let lane = warp::lane_id() as usize;
        let warp_in_block = thread::threadIdx_x() / ROW_SIZE as u32;
        let warps_per_block = thread::blockDim_x() / ROW_SIZE as u32;
        let row = thread::blockIdx_x() * warps_per_block + warp_in_block;

        if row < row_count {
            let index = row as usize * ROW_SIZE + lane;
            let value = x[index];
            let mean = warp_sum_f32(value) / ROW_SIZE_F32;
            let centered = value - mean;
            let variance = warp_sum_f32(centered * centered) / ROW_SIZE_F32;
            let inv_std = 1.0 / sqrt_f32(variance + epsilon);
            let normalized = centered * inv_std;

            unsafe {
                *out.get_unchecked_mut(index) = fma_f32(normalized, gamma[lane], beta[lane]);
            }
        }
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn gpt_layer_norm_kernel(
        residual: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        weight_global_scale: &[f32],
        bias_global_scale: &[f32],
        mut normalized: DisjointSlice<f32>,
        mut normalized_amax: DisjointSlice<f32>,
        mut mean_out: DisjointSlice<f32>,
        mut inv_std_out: DisjointSlice<f32>,
        row_count: u32,
        embedding_dim: u32,
        epsilon: f32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { GPT_LAYER_NORM_WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / WARP_SIZE;

        if row < row_count {
            let row_base = row as usize * embedding_dim as usize;
            let cols = layer_norm_columns3!(thread, GPT_LAYER_NORM_THREADS_PER_BLOCK);
            let values = layer_norm_map3!(cols, |col| f32_column(
                residual,
                row_base,
                col,
                embedding_dim
            ));

            let mean = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                layer_norm_sum3!(values),
                lane,
                warp_in_block,
                warp_sum_f32
            ) / embedding_dim as f32;
            layer_norm_store_row!(&mut mean_out, row, lane, warp_in_block, mean);
            let centered = layer_norm_map3_indexed!(cols, |index, col| centered_column(
                col,
                embedding_dim,
                values[index],
                mean
            ));
            let variance_sum = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                layer_norm_square_sum3!(centered),
                lane,
                warp_in_block,
                warp_sum_f32
            );
            let inv_std = 1.0 / sqrt_f32(variance_sum / embedding_dim as f32 + epsilon);
            layer_norm_store_row!(&mut inv_std_out, row, lane, warp_in_block, inv_std);
            let normalized_values =
                layer_norm_map3_indexed!(cols, |index, col| nvfp4_affine_normalized_column(
                    weight_bytes,
                    weight_scales,
                    bias_bytes,
                    bias_scales,
                    col,
                    embedding_dim,
                    centered[index],
                    inv_std,
                    weight_global_scale[0],
                    bias_global_scale[0],
                ));

            layer_norm_store3!(
                &mut normalized,
                row_base,
                cols,
                embedding_dim,
                normalized_values
            );

            let local_amax = max_abs3(
                normalized_values[0],
                normalized_values[1],
                normalized_values[2],
            );
            let block_amax = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                local_amax,
                lane,
                warp_in_block,
                warp_max_f32
            );

            layer_norm_store_row!(&mut normalized_amax, row, lane, warp_in_block, block_amax);
        }
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn gpt_layer_norm_save_residual_f16_kernel(
        residual: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        weight_global_scale: &[f32],
        bias_global_scale: &[f32],
        mut normalized: DisjointSlice<f32>,
        mut normalized_amax: DisjointSlice<f32>,
        mut mean_out: DisjointSlice<f32>,
        mut inv_std_out: DisjointSlice<f32>,
        mut residual_f16: DisjointSlice<u16>,
        row_count: u32,
        embedding_dim: u32,
        epsilon: f32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { GPT_LAYER_NORM_WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / WARP_SIZE;

        if row < row_count {
            let row_base = row as usize * embedding_dim as usize;
            let cols = layer_norm_columns3!(thread, GPT_LAYER_NORM_THREADS_PER_BLOCK);
            let values = layer_norm_map3!(cols, |col| f32_column(
                residual,
                row_base,
                col,
                embedding_dim
            ));

            layer_norm_store_f16_3!(&mut residual_f16, row_base, cols, embedding_dim, values);

            let mean = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                layer_norm_sum3!(values),
                lane,
                warp_in_block,
                warp_sum_f32
            ) / embedding_dim as f32;
            layer_norm_store_row!(&mut mean_out, row, lane, warp_in_block, mean);
            let centered = layer_norm_map3_indexed!(cols, |index, col| centered_column(
                col,
                embedding_dim,
                values[index],
                mean
            ));
            let variance_sum = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                layer_norm_square_sum3!(centered),
                lane,
                warp_in_block,
                warp_sum_f32
            );
            let inv_std = 1.0 / sqrt_f32(variance_sum / embedding_dim as f32 + epsilon);
            layer_norm_store_row!(&mut inv_std_out, row, lane, warp_in_block, inv_std);
            let normalized_values =
                layer_norm_map3_indexed!(cols, |index, col| nvfp4_affine_normalized_column(
                    weight_bytes,
                    weight_scales,
                    bias_bytes,
                    bias_scales,
                    col,
                    embedding_dim,
                    centered[index],
                    inv_std,
                    weight_global_scale[0],
                    bias_global_scale[0],
                ));

            layer_norm_store3!(
                &mut normalized,
                row_base,
                cols,
                embedding_dim,
                normalized_values
            );

            let local_amax = max_abs3(
                normalized_values[0],
                normalized_values[1],
                normalized_values[2],
            );
            let block_amax = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                local_amax,
                lane,
                warp_in_block,
                warp_max_f32
            );

            layer_norm_store_row!(&mut normalized_amax, row, lane, warp_in_block, block_amax);
        }
    }
}
