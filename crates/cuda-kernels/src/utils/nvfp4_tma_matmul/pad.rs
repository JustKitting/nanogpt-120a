use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::launch::linear_config;

const THREADS_PER_BLOCK: u32 = 256;

pub struct F32PadArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub output: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub cols: u32,
    pub padded_rows: u32,
    pub padded_cols: u32,
}

pub struct F32CropArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub output: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub cols: u32,
    pub input_cols: u32,
}

pub struct U4RowPadArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<u8>,
    pub output: &'out mut DeviceBuffer<u8>,
    pub rows: u32,
    pub padded_rows: u32,
    pub cols_u4: u32,
}

pub struct TmaMatrixPadModule {
    module: module::LoadedModule,
}

impl TmaMatrixPadModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: module::from_module(module)?,
        })
    }

    pub fn pad_f32(&self, args: F32PadArgs<'_, '_>) -> Result<(), DriverError> {
        assert!(args.input.len() >= args.rows as usize * args.cols as usize);
        assert!(args.output.len() >= args.padded_rows as usize * args.padded_cols as usize);
        self.module.f32_pad_matrix_kernel(
            args.stream,
            linear_config(args.padded_rows * args.padded_cols, THREADS_PER_BLOCK),
            args.input,
            args.output,
            args.rows,
            args.cols,
            args.padded_rows,
            args.padded_cols,
        )
    }

    pub fn transpose_pad_f32(&self, args: F32PadArgs<'_, '_>) -> Result<(), DriverError> {
        assert!(args.input.len() >= args.rows as usize * args.cols as usize);
        assert!(args.output.len() >= args.padded_rows as usize * args.padded_cols as usize);
        self.module.f32_transpose_pad_matrix_kernel(
            args.stream,
            linear_config(args.padded_rows * args.padded_cols, THREADS_PER_BLOCK),
            args.input,
            args.output,
            args.rows,
            args.cols,
            args.padded_rows,
            args.padded_cols,
        )
    }

    pub fn crop_f32(&self, args: F32CropArgs<'_, '_>) -> Result<(), DriverError> {
        assert!(args.input.len() >= args.rows as usize * args.input_cols as usize);
        assert!(args.output.len() >= args.rows as usize * args.cols as usize);
        self.module.f32_crop_cols_kernel(
            args.stream,
            linear_config(args.rows * args.cols, THREADS_PER_BLOCK),
            args.input,
            args.output,
            args.rows,
            args.cols,
            args.input_cols,
        )
    }

    pub fn pad_u4_rows(&self, args: U4RowPadArgs<'_, '_>) -> Result<(), DriverError> {
        assert!(args.cols_u4.is_multiple_of(2));
        let row_bytes = args.cols_u4 / 2;
        assert!(args.input.len() >= args.rows as usize * row_bytes as usize);
        assert!(args.output.len() >= args.padded_rows as usize * row_bytes as usize);
        self.module.u4_pad_rows_kernel(
            args.stream,
            linear_config(args.padded_rows * row_bytes, THREADS_PER_BLOCK),
            args.input,
            args.output,
            args.rows,
            args.padded_rows,
            row_bytes,
        )
    }
}

#[cuda_module]
mod module {
    use super::*;

    #[kernel]
    pub fn f32_pad_matrix_kernel(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        padded_rows: u32,
        padded_cols: u32,
    ) {
        let stride = thread::gridDim_x() * thread::blockDim_x();
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let len = padded_rows * padded_cols;
        while index < len {
            let row = index / padded_cols;
            let col = index - row * padded_cols;
            let value = if row < rows && col < cols {
                input[(row * cols + col) as usize]
            } else {
                0.0
            };
            unsafe {
                *output.get_unchecked_mut(index as usize) = value;
            }
            index += stride;
        }
    }

    #[kernel]
    pub fn f32_transpose_pad_matrix_kernel(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        padded_rows: u32,
        padded_cols: u32,
    ) {
        let stride = thread::gridDim_x() * thread::blockDim_x();
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let len = padded_rows * padded_cols;
        while index < len {
            let row = index / padded_cols;
            let col = index - row * padded_cols;
            let value = if row < cols && col < rows {
                input[(col * cols + row) as usize]
            } else {
                0.0
            };
            unsafe {
                *output.get_unchecked_mut(index as usize) = value;
            }
            index += stride;
        }
    }

    #[kernel]
    pub fn f32_crop_cols_kernel(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        input_cols: u32,
    ) {
        let stride = thread::gridDim_x() * thread::blockDim_x();
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let len = rows * cols;
        while index < len {
            let row = index / cols;
            let col = index - row * cols;
            unsafe {
                *output.get_unchecked_mut(index as usize) =
                    input[(row * input_cols + col) as usize];
            }
            index += stride;
        }
    }

    #[kernel]
    pub fn u4_pad_rows_kernel(
        input: &[u8],
        mut output: DisjointSlice<u8>,
        rows: u32,
        padded_rows: u32,
        row_bytes: u32,
    ) {
        let stride = thread::gridDim_x() * thread::blockDim_x();
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let len = padded_rows * row_bytes;
        while index < len {
            let row = index / row_bytes;
            let col = index - row * row_bytes;
            let value = if row < rows {
                input[(row * row_bytes + col) as usize]
            } else {
                0
            };
            unsafe {
                *output.get_unchecked_mut(index as usize) = value;
            }
            index += stride;
        }
    }
}
