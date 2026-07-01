use cuda_core::DriverError;

use super::super::args::RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs;
use super::super::launcher::Nvfp4QuantModule;
use super::super::shape::MsEdenPackGrid;
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

impl Nvfp4QuantModule {
    pub(super) fn launch_rowwise_transpose_fallback_no_chunk_amax(
        &self,
        args: &mut RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
        pack: MsEdenPackGrid,
    ) -> Result<(), DriverError> {
        self.ms_eden_rowwise_transpose
            .padded
            .rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
                args.stream,
                pack.config(),
                args.input.bytes,
                args.input.scales,
                args.input.global_scales,
                &mut *args.out_fp4,
                &mut *args.out_scales,
                &mut *args.out_global_scales,
                &*args.out_global_scale,
                pack.chunk_count,
                args.source_rows,
                args.source_cols,
                args.dst_row_len,
                QUARTET_MS_EDEN_SCALE_OVERRIDE,
                args.sign_seed,
                args.scale_seed,
            )
    }
}
