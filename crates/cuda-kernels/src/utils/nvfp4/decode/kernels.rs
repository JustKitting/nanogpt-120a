use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use super::DECODE_THREADS_PER_BLOCK;
use crate::nvfp4::{nvfp4_rowwise_value, nvfp4_value};

#[cuda_module]
mod module {
    use super::*;

    #[kernel]
    pub fn nvfp4_decode_transpose_f32_kernel(
        bytes: &[u8],
        scales: &[u8],
        global_scale: &[f32],
        mut output: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
    ) {
        let index = thread::blockIdx_x() * DECODE_THREADS_PER_BLOCK + thread::threadIdx_x();
        let len = rows * cols;
        if index < len {
            let value = nvfp4_value(bytes, scales, global_scale[0], index as usize);
            let out_index = (index % cols) * rows + index / cols;

            unsafe {
                *output.get_unchecked_mut(out_index as usize) = value;
            }
        }
    }

    #[kernel]
    pub fn nvfp4_decode_rowwise_transpose_f32_kernel(
        bytes: &[u8],
        scales: &[u8],
        global_scales: &[f32],
        mut output: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
    ) {
        let index = thread::blockIdx_x() * DECODE_THREADS_PER_BLOCK + thread::threadIdx_x();
        let len = rows * cols;
        if index < len {
            let row = index / cols;
            let col = index - row * cols;
            let value = nvfp4_rowwise_value(
                bytes,
                scales,
                global_scales,
                cols as usize,
                row as usize,
                col as usize,
            );
            let out_index = col * rows + row;

            unsafe {
                *output.get_unchecked_mut(out_index as usize) = value;
            }
        }
    }
}

pub(super) use module::{LoadedModule, from_module};
