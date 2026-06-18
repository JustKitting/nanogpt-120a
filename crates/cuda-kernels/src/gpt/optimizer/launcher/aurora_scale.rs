use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use super::super::threads::MATRIX_THREADS_PER_BLOCK;
use super::{OptimizerModule, matrix_config};

impl OptimizerModule {
    pub fn row_inv_norm(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        row_scale: &mut DeviceBuffer<f32>,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> Result<(), DriverError> {
        self.apply.aurora.row_balance.row_inv_norm_kernel(
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

    pub fn row_scale_apply(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        row_scale: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        rows: u32,
        cols: u32,
    ) -> Result<(), DriverError> {
        self.apply.aurora.row_balance.row_scale_apply_kernel(
            stream,
            matrix_config(rows * cols),
            x,
            row_scale,
            out,
            rows,
            cols,
        )
    }
}
