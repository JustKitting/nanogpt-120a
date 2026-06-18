use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use super::super::threads::MATRIX_THREADS_PER_BLOCK;
use super::{OptimizerModule, matrix_config};
use crate::optimizer::polar_normalize_chunks;

impl OptimizerModule {
    pub fn polar_normalize(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        chunks: &mut DeviceBuffer<f32>,
        inv_norm: &mut DeviceBuffer<f32>,
        len: u32,
    ) -> Result<(), DriverError> {
        let chunk_count = self.polar_inv_norm(stream, x, chunks, inv_norm, len)?;
        self.apply.aurora.polar.polar_scale_from_inv_norm_kernel(
            stream,
            matrix_config(len),
            x,
            out,
            &*inv_norm,
            len,
        )?;
        debug_assert_eq!(chunk_count as usize, polar_normalize_chunks(len as usize));
        Ok(())
    }

    pub fn polar_normalize_in_place(
        &self,
        stream: &CudaStream,
        x: &mut DeviceBuffer<f32>,
        chunks: &mut DeviceBuffer<f32>,
        inv_norm: &mut DeviceBuffer<f32>,
        len: u32,
    ) -> Result<(), DriverError> {
        self.polar_inv_norm(stream, &*x, chunks, inv_norm, len)?;
        self.apply
            .aurora
            .polar
            .polar_scale_in_place_from_inv_norm_kernel(
                stream,
                matrix_config(len),
                x,
                &*inv_norm,
                len,
            )
    }

    fn polar_inv_norm(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        chunks: &mut DeviceBuffer<f32>,
        inv_norm: &mut DeviceBuffer<f32>,
        len: u32,
    ) -> Result<u32, DriverError> {
        let chunk_count = polar_normalize_chunks(len as usize) as u32;
        self.apply.aurora.polar.polar_chunk_sum_kernel(
            stream,
            LaunchConfig {
                grid_dim: (chunk_count, 1, 1),
                block_dim: (MATRIX_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            x,
            chunks,
            len,
        )?;
        self.apply.aurora.polar.polar_inv_norm_from_chunks_kernel(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (MATRIX_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            &*chunks,
            inv_norm,
            chunk_count,
        )?;
        Ok(chunk_count)
    }

    pub fn elementwise_linear_combination(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        a_scale: f32,
        b_scale: f32,
        len: u32,
    ) -> Result<(), DriverError> {
        self.apply
            .aurora
            .elementwise
            .elementwise_linear_combination_kernel(
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
}
