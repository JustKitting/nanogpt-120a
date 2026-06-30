use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::args::{
    Nvfp4TransposeMsEdenDeviceScaleQuantArgs, QuartetBackwardMsEdenDeviceScaleQuantArgs,
    RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs,
};
use super::launcher::Nvfp4QuantModule;
use super::shape::{grid_config, tensor_amax_chunk_count};

impl Nvfp4QuantModule {
    pub fn quartet_backward_ms_eden_global_scale_from_chunks(
        &self,
        stream: &CudaStream,
        chunk_amax: &DeviceBuffer<f32>,
        out_global_scale: &mut DeviceBuffer<f32>,
        chunk_count: u32,
    ) -> Result<(), DriverError> {
        self.ms_eden
            .quartet_backward_ms_eden_global_scale_from_chunks_kernel(
                stream,
                grid_config(1),
                chunk_amax,
                out_global_scale,
                chunk_count,
            )
    }

    pub(super) fn derive_rowwise_nvfp4_transpose_global_scale(
        &self,
        args: &mut RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let chunk_count = tensor_amax_chunk_count(args.source_rows * args.source_cols);
        self.ms_eden.rowwise_nvfp4_chunk_amax_kernel(
            args.stream,
            grid_config(chunk_count),
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.out_chunk_amax,
            args.source_rows,
            args.source_cols,
        )?;

        self.quartet_backward_ms_eden_global_scale_from_chunks(
            args.stream,
            &*args.out_chunk_amax,
            &mut *args.out_global_scale,
            chunk_count,
        )
    }

    pub(super) fn derive_nvfp4_transpose_global_scale(
        &self,
        args: &mut Nvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let element_count = args.source_rows * args.source_cols;
        let chunk_count = tensor_amax_chunk_count(element_count);
        self.ms_eden.nvfp4_chunk_amax_kernel(
            args.stream,
            grid_config(chunk_count),
            args.input.bytes,
            args.input.scales,
            args.input.global_scale,
            args.out_chunk_amax,
            element_count,
        )?;

        self.quartet_backward_ms_eden_global_scale_from_chunks(
            args.stream,
            &*args.out_chunk_amax,
            &mut *args.out_global_scale,
            chunk_count,
        )
    }

    pub(super) fn derive_fp32_quartet_backward_ms_eden_global_scale(
        &self,
        args: &mut QuartetBackwardMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let chunk_count = self.tensor_chunk_amax_f32(
            args.stream,
            args.x,
            &mut *args.out_chunk_amax,
            args.row_count * args.src_row_len,
        )?;

        self.quartet_backward_ms_eden_global_scale_from_chunks(
            args.stream,
            &*args.out_chunk_amax,
            &mut *args.out_global_scale,
            chunk_count,
        )
    }
}
