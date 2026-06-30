use cuda_core::DriverError;

use super::args::Nvfp4TransposeMsEdenDeviceScaleQuantArgs;
use super::launcher::Nvfp4QuantModule;
use super::shape::MsEdenPackGrid;
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

impl Nvfp4QuantModule {
    pub fn nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
        &self,
        mut args: Nvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_nvfp4_transpose_global_scale(&mut args)?;
        let pack = MsEdenPackGrid::for_elements(args.source_cols * args.dst_row_len);
        self.ms_eden_nvfp4_transpose
            .nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel(
                args.stream,
                pack.config(),
                args.input.bytes,
                args.input.scales,
                args.input.global_scale,
                args.out_fp4,
                args.out_scales,
                args.out_global_scales,
                args.out_chunk_amax,
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

    pub fn nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        &self,
        mut args: Nvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_nvfp4_transpose_global_scale(&mut args)?;
        let pack = MsEdenPackGrid::for_elements(args.source_cols * args.dst_row_len);
        if pack.is_exact() {
            return self
                .ms_eden_nvfp4_transpose
                .nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
                    args.stream,
                    pack.config(),
                    args.input.bytes,
                    args.input.scales,
                    args.input.global_scale,
                    args.out_fp4,
                    args.out_scales,
                    args.out_global_scales,
                    &*args.out_global_scale,
                    args.source_rows,
                    args.source_cols,
                    args.dst_row_len,
                    QUARTET_MS_EDEN_SCALE_OVERRIDE,
                    args.sign_seed,
                    args.scale_seed,
                );
        }

        self.ms_eden_nvfp4_transpose
            .nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
                args.stream,
                pack.config(),
                args.input.bytes,
                args.input.scales,
                args.input.global_scale,
                args.out_fp4,
                args.out_scales,
                args.out_global_scales,
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
