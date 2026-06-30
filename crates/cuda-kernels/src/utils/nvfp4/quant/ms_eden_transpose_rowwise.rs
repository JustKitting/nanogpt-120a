use cuda_core::DriverError;

use super::args::RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs;
use super::launcher::Nvfp4QuantModule;
use super::shape::{MsEdenPackGrid, RowwiseTransposeNoPad};
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

impl Nvfp4QuantModule {
    pub fn rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
        &self,
        mut args: RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_rowwise_nvfp4_transpose_global_scale(&mut args)?;
        let pack = MsEdenPackGrid::for_elements(args.source_cols * args.dst_row_len);
        self.ms_eden_rowwise_transpose
            .rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel(
                args.stream,
                pack.config(),
                args.input.bytes,
                args.input.scales,
                args.input.global_scales,
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

    pub fn rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        &self,
        mut args: RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_rowwise_nvfp4_transpose_global_scale(&mut args)?;
        let pack = MsEdenPackGrid::for_elements(args.source_cols * args.dst_row_len);
        if pack.is_exact() {
            if let Some(no_pad) =
                RowwiseTransposeNoPad::new(args.source_rows, args.source_cols, args.dst_row_len)
            {
                if let Some(source_cols_shift) = no_pad.source_cols_shift() {
                    return self
                        .ms_eden_rowwise_transpose
                        .rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_source_cols_pow2_kernel(
                            args.stream,
                            pack.config(),
                            args.input.bytes,
                            args.input.scales,
                            args.input.global_scales,
                            args.out_fp4,
                            args.out_scales,
                            args.out_global_scales,
                            &*args.out_global_scale,
                            source_cols_shift,
                            no_pad.chunks_per_row_shift,
                            QUARTET_MS_EDEN_SCALE_OVERRIDE,
                            args.sign_seed,
                            args.scale_seed,
                        );
                }

                return self
                    .ms_eden_rowwise_transpose
                    .rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_kernel(
                        args.stream,
                        pack.config(),
                        args.input.bytes,
                        args.input.scales,
                        args.input.global_scales,
                        args.out_fp4,
                        args.out_scales,
                        args.out_global_scales,
                        &*args.out_global_scale,
                        no_pad.source_cols,
                        no_pad.chunks_per_row_shift,
                        QUARTET_MS_EDEN_SCALE_OVERRIDE,
                        args.sign_seed,
                        args.scale_seed,
                    );
            }

            return self
                .ms_eden_rowwise_transpose
                .rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
                    args.stream,
                    pack.config(),
                    args.input.bytes,
                    args.input.scales,
                    args.input.global_scales,
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

        self.ms_eden_rowwise_transpose
            .rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
                args.stream,
                pack.config(),
                args.input.bytes,
                args.input.scales,
                args.input.global_scales,
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
