use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use super::super::kernels::MATRIX_THREADS_PER_BLOCK;
use super::{OptimizerModule, matrix_config};

impl OptimizerModule {
    pub fn row_scale_apply(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        row_scale: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        rows: u32,
        cols: u32,
    ) -> Result<(), DriverError> {
        self.apply.row.row_scale_apply_kernel(
            stream,
            matrix_config(rows * cols),
            x,
            row_scale,
            out,
            rows,
            cols,
        )
    }

    pub fn row_scale_refine(
        &self,
        stream: &CudaStream,
        u: &DeviceBuffer<f32>,
        row_scale: &mut DeviceBuffer<f32>,
        rows: u32,
        cols: u32,
        target_row_sq: f32,
        eps: f32,
    ) -> Result<(), DriverError> {
        self.apply.row.row_scale_refine_kernel(
            stream,
            LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (MATRIX_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            u,
            row_scale,
            rows,
            cols,
            target_row_sq,
            eps,
        )
    }
}
