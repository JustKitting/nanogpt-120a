use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::{OptimizerModule, matrix_config};

impl OptimizerModule {
    pub fn aurora_momentum_orient(
        &self,
        stream: &CudaStream,
        grad: &DeviceBuffer<f32>,
        momentum: &mut DeviceBuffer<f32>,
        oriented: &mut DeviceBuffer<f32>,
        mu: f32,
        rows: u32,
        cols: u32,
    ) -> Result<(u32, u32, bool), DriverError> {
        let len = rows * cols;
        if rows < cols {
            self.apply
                .aurora
                .momentum
                .aurora_momentum_orient_transpose_kernel(
                    stream,
                    matrix_config(len),
                    grad,
                    momentum,
                    oriented,
                    mu,
                    rows,
                    cols,
                )?;
            return Ok((cols, rows, true));
        }

        self.apply.aurora.momentum.aurora_momentum_orient_kernel(
            stream,
            matrix_config(len),
            grad,
            momentum,
            oriented,
            mu,
            len,
        )?;
        Ok((rows, cols, false))
    }
}
