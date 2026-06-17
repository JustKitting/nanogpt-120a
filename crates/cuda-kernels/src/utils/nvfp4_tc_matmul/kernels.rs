use cuda_device::{DisjointSlice, cuda_module, kernel};

use crate::mma::{Nvfp4ProjectionParams, nvfp4_projection_nobias_kernel_body};

pub const PAD_THREADS_PER_BLOCK: u32 = 256;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn nvfp4_tc_matmul_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        mut out: DisjointSlice<f32>,
        params: Nvfp4ProjectionParams,
    ) {
        nvfp4_projection_nobias_kernel_body(
            input_bytes,
            input_scales,
            input_global_scales,
            weight_bytes,
            weight_scales,
            &mut out,
            params,
        );
    }

    #[kernel]
    pub fn fp32_pad_rows_kernel(
        src: &[f32],
        mut dst: DisjointSlice<f32>,
        rows: u32,
        src_cols: u32,
        dst_cols: u32,
    ) {
        let index = cuda_device::thread::blockIdx_x() * PAD_THREADS_PER_BLOCK
            + cuda_device::thread::threadIdx_x();
        if index < rows * dst_cols {
            let row = index / dst_cols;
            let col = index - row * dst_cols;
            let value = if col < src_cols {
                src[(row * src_cols + col) as usize]
            } else {
                0.0
            };

            unsafe {
                *dst.get_unchecked_mut(index as usize) = value;
            }
        }
    }
}
