use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::args::{RowAmaxArgs, TensorAmaxArgs};
use super::launcher::Nvfp4QuantModule;
use super::shape::{grid_config, tensor_amax_chunk_count};

impl Nvfp4QuantModule {
    pub fn row_amax_f32(&self, args: RowAmaxArgs<'_, '_>) -> Result<(), DriverError> {
        self.row_amax.row_amax_f32_kernel(
            args.stream,
            grid_config(args.row_count),
            args.x,
            args.out,
            args.row_count,
            args.row_len,
        )
    }

    pub fn tensor_amax_f32(&self, args: TensorAmaxArgs<'_, '_>) -> Result<(), DriverError> {
        let chunk_count =
            self.tensor_chunk_amax_f32(args.stream, args.x, args.chunk_amax, args.element_count)?;
        self.tensor_amax_from_chunks_f32(args.stream, &*args.chunk_amax, args.out, chunk_count)
    }

    pub fn tensor_amax_from_chunks_f32(
        &self,
        stream: &CudaStream,
        chunk_amax: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        chunk_count: u32,
    ) -> Result<(), DriverError> {
        self.row_amax.tensor_amax_from_chunks_f32_kernel(
            stream,
            grid_config(1),
            chunk_amax,
            out,
            chunk_count,
        )
    }

    pub(super) fn tensor_chunk_amax_f32(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        element_count: u32,
    ) -> Result<u32, DriverError> {
        let chunk_count = tensor_amax_chunk_count(element_count);
        self.row_amax.tensor_chunk_amax_f32_kernel(
            stream,
            grid_config(chunk_count),
            x,
            out,
            element_count,
        )?;
        Ok(chunk_count)
    }
}
