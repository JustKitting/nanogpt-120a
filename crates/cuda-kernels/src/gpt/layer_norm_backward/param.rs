use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::layer_norm_reduce::{layer_norm_block_reduce, layer_norm_store_row};
use crate::layer_norm_utils::{f16_column, f32_column};
use crate::warp_reduce::{thread_lane_warp, warp_sum_f32};

pub const PARAM_THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const WARPS_PER_BLOCK: u32 = PARAM_THREADS_PER_BLOCK / WARP_SIZE;
const ROWS_PER_THREAD: u32 = 4;
const UNROLLED_ROW_STRIDE: u32 = PARAM_THREADS_PER_BLOCK * ROWS_PER_THREAD;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
pub(super) mod kernels {
    use super::*;

    macro_rules! layer_norm_backward_params_body {
        (
            $residual_column:path;
            $residual:ident $d_normalized:ident $mean:ident $inv_std:ident;
            $d_weight:ident $d_bias:ident $row_count:ident $embedding_dim:ident
        ) => {{
            static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
                SharedArray::UNINIT;

            let col = thread::blockIdx_x();
            let (tid, lane, warp_in_block) = thread_lane_warp();

            if col < $embedding_dim {
                let mut weight_local = 0.0f32;
                let mut bias_local = 0.0f32;
                let mut row = tid;

                macro_rules! accumulate_param_grad {
                    ($weight:ident, $bias:ident, $row:expr) => {{
                        let row = $row;
                        let offset = row as usize * $embedding_dim as usize + col as usize;
                        let grad = $d_normalized[offset];
                        let row_base = row as usize * $embedding_dim as usize;
                        let xhat = ($residual_column($residual, row_base, col, $embedding_dim)
                            - $mean[row as usize])
                            * $inv_std[row as usize];
                        $weight += grad * xhat;
                        $bias += grad;
                    }};
                }

                while row + PARAM_THREADS_PER_BLOCK * 3 < $row_count {
                    accumulate_param_grad!(weight_local, bias_local, row);
                    accumulate_param_grad!(weight_local, bias_local, row + PARAM_THREADS_PER_BLOCK);
                    accumulate_param_grad!(weight_local, bias_local, row + PARAM_THREADS_PER_BLOCK * 2);
                    accumulate_param_grad!(weight_local, bias_local, row + PARAM_THREADS_PER_BLOCK * 3);
                    row += UNROLLED_ROW_STRIDE;
                }

                while row < $row_count {
                    accumulate_param_grad!(weight_local, bias_local, row);
                    row += PARAM_THREADS_PER_BLOCK;
                }

                let weight_sum = layer_norm_block_reduce!(WARP_SUMS, WARPS_PER_BLOCK, weight_local, lane, warp_in_block, warp_sum_f32);
                let bias_sum = layer_norm_block_reduce!(WARP_SUMS, WARPS_PER_BLOCK, bias_local, lane, warp_in_block, warp_sum_f32);

                layer_norm_store_row!(&mut $d_weight, col, lane, warp_in_block, weight_sum);
                layer_norm_store_row!(&mut $d_bias, col, lane, warp_in_block, bias_sum);
            }
        }};
    }

    #[kernel]
    pub fn layer_norm_backward_params_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        mut d_weight: DisjointSlice<f32>,
        mut d_bias: DisjointSlice<f32>,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_params_body!(
            f16_column;
            residual d_normalized mean inv_std;
            d_weight d_bias row_count embedding_dim
        );
    }

    #[kernel]
    pub fn layer_norm_backward_params_f32_kernel(
        residual: &[f32],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        mut d_weight: DisjointSlice<f32>,
        mut d_bias: DisjointSlice<f32>,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_params_body!(
            f32_column;
            residual d_normalized mean inv_std;
            d_weight d_bias row_count embedding_dim
        );
    }
}
