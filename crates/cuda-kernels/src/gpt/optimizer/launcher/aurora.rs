use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use super::super::threads::MATRIX_THREADS_PER_BLOCK;
use super::{OptimizerModule, matrix_config};

impl OptimizerModule {
    pub fn aurora_momentum(
        &self,
        stream: &CudaStream,
        grad: &DeviceBuffer<f32>,
        momentum: &mut DeviceBuffer<f32>,
        update: &mut DeviceBuffer<f32>,
        mu: f32,
        len: u32,
    ) -> Result<(), DriverError> {
        self.apply.aurora.momentum.aurora_momentum_kernel(
            stream,
            matrix_config(len),
            grad,
            momentum,
            update,
            mu,
            len,
        )
    }

    pub fn matrix_frobenius_norm(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        len: u32,
    ) -> Result<(), DriverError> {
        self.apply.aurora.matrix.matrix_frobenius_norm_kernel(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (MATRIX_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            x,
            out,
            len,
        )
    }

    pub fn matrix_scale_in_place(
        &self,
        stream: &CudaStream,
        x: &mut DeviceBuffer<f32>,
        norm: &DeviceBuffer<f32>,
        len: u32,
    ) -> Result<(), DriverError> {
        self.apply.aurora.matrix.matrix_scale_in_place_kernel(
            stream,
            matrix_config(len),
            x,
            norm,
            len,
        )
    }

    pub fn matrix_combine(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        a_scale: f32,
        b_scale: f32,
        len: u32,
    ) -> Result<(), DriverError> {
        self.apply.aurora.matrix.matrix_combine_kernel(
            stream,
            matrix_config(len),
            a,
            b,
            out,
            a_scale,
            b_scale,
            len,
        )
    }

    pub fn row_inv_norm(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        row_scale: &mut DeviceBuffer<f32>,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> Result<(), DriverError> {
        self.apply.aurora.row.row_inv_norm_kernel(
            stream,
            LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (MATRIX_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            x,
            row_scale,
            rows,
            cols,
            eps,
        )
    }
}
