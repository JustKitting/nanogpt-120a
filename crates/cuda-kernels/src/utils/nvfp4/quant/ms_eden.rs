use cuda_core::DriverError;

use super::args::{
    MsEdenDeviceScaleQuantArgs, MsEdenQuantArgs, MsEdenTransposeDeviceScaleQuantArgs,
    QuartetBackwardMsEdenDeviceScaleQuantArgs, QuartetBackwardMsEdenQuantArgs,
};
use super::launcher::Nvfp4QuantModule;
use super::shape::MsEdenPackGrid;
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

impl Nvfp4QuantModule {
    pub fn fp32_to_nvfp4_ms_eden(&self, args: MsEdenQuantArgs<'_, '_>) -> Result<(), DriverError> {
        let pack = MsEdenPackGrid::for_elements(args.row_count * args.dst_row_len);
        self.ms_eden.fp32_to_nvfp4_ms_eden_kernel(
            args.stream,
            pack.config(),
            args.x,
            args.out_fp4,
            args.out_scales,
            args.out_global_scales,
            args.out_chunk_amax,
            pack.chunk_count,
            args.src_row_len,
            args.dst_row_len,
            args.global_scale,
            args.scale_override,
            args.sign_seed,
            args.scale_seed,
        )
    }

    pub fn fp32_to_nvfp4_ms_eden_device_scale(
        &self,
        args: MsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let pack = MsEdenPackGrid::for_elements(args.row_count * args.dst_row_len);
        self.ms_eden.fp32_to_nvfp4_ms_eden_device_scale_kernel(
            args.stream,
            pack.config(),
            args.x,
            args.out_fp4,
            args.out_scales,
            args.out_global_scales,
            args.out_chunk_amax,
            args.global_scale,
            pack.chunk_count,
            args.src_row_len,
            args.dst_row_len,
            args.scale_override,
            args.sign_seed,
            args.scale_seed,
        )
    }

    pub fn fp32_to_nvfp4_ms_eden_device_scale_no_chunk_amax(
        &self,
        args: MsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let pack = MsEdenPackGrid::for_elements(args.row_count * args.dst_row_len);
        if pack.is_exact() {
            return self
                .ms_eden
                .fp32_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
                    args.stream,
                    pack.config(),
                    args.x,
                    args.out_fp4,
                    args.out_scales,
                    args.out_global_scales,
                    args.global_scale,
                    args.src_row_len,
                    args.dst_row_len,
                    args.scale_override,
                    args.sign_seed,
                    args.scale_seed,
                );
        }

        self.ms_eden
            .fp32_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
                args.stream,
                pack.config(),
                args.x,
                args.out_fp4,
                args.out_scales,
                args.out_global_scales,
                args.global_scale,
                pack.chunk_count,
                args.src_row_len,
                args.dst_row_len,
                args.scale_override,
                args.sign_seed,
                args.scale_seed,
            )
    }

    pub fn fp32_transpose_to_nvfp4_ms_eden_device_scale(
        &self,
        args: MsEdenTransposeDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let pack = MsEdenPackGrid::for_elements(args.source_cols * args.dst_row_len);
        self.ms_eden
            .fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel(
                args.stream,
                pack.config(),
                args.x,
                args.out_fp4,
                args.out_scales,
                args.out_global_scales,
                args.out_chunk_amax,
                args.global_scale,
                pack.chunk_count,
                args.source_rows,
                args.source_cols,
                args.dst_row_len,
                args.scale_override,
                args.sign_seed,
                args.scale_seed,
            )
    }

    pub fn fp32_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax(
        &self,
        args: MsEdenTransposeDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let pack = MsEdenPackGrid::for_elements(args.source_cols * args.dst_row_len);
        if pack.is_exact() {
            return self
                .ms_eden
                .fp32_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
                    args.stream,
                    pack.config(),
                    args.x,
                    args.out_fp4,
                    args.out_scales,
                    args.out_global_scales,
                    args.global_scale,
                    args.source_rows,
                    args.source_cols,
                    args.dst_row_len,
                    args.scale_override,
                    args.sign_seed,
                    args.scale_seed,
                );
        }

        self.ms_eden
            .fp32_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
                args.stream,
                pack.config(),
                args.x,
                args.out_fp4,
                args.out_scales,
                args.out_global_scales,
                args.global_scale,
                pack.chunk_count,
                args.source_rows,
                args.source_cols,
                args.dst_row_len,
                args.scale_override,
                args.sign_seed,
                args.scale_seed,
            )
    }

    pub fn fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale(
        &self,
        mut args: QuartetBackwardMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_fp32_quartet_backward_ms_eden_global_scale(&mut args)?;
        self.fp32_to_nvfp4_ms_eden_device_scale(MsEdenDeviceScaleQuantArgs {
            stream: args.stream,
            x: args.x,
            out_fp4: args.out_fp4,
            out_scales: args.out_scales,
            out_global_scales: args.out_global_scales,
            out_chunk_amax: args.out_chunk_amax,
            global_scale: &*args.out_global_scale,
            row_count: args.row_count,
            src_row_len: args.src_row_len,
            dst_row_len: args.dst_row_len,
            scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
            sign_seed: args.sign_seed,
            scale_seed: args.scale_seed,
        })
    }

    pub fn fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        &self,
        mut args: QuartetBackwardMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_fp32_quartet_backward_ms_eden_global_scale(&mut args)?;
        self.fp32_to_nvfp4_ms_eden_device_scale_no_chunk_amax(MsEdenDeviceScaleQuantArgs {
            stream: args.stream,
            x: args.x,
            out_fp4: args.out_fp4,
            out_scales: args.out_scales,
            out_global_scales: args.out_global_scales,
            out_chunk_amax: args.out_chunk_amax,
            global_scale: &*args.out_global_scale,
            row_count: args.row_count,
            src_row_len: args.src_row_len,
            dst_row_len: args.dst_row_len,
            scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
            sign_seed: args.sign_seed,
            scale_seed: args.scale_seed,
        })
    }

    pub fn fp32_to_nvfp4_quartet_backward_ms_eden_with_global_scale(
        &self,
        args: QuartetBackwardMsEdenQuantArgs<'_, '_>,
        global_scale: f32,
    ) -> Result<f32, DriverError> {
        self.fp32_to_nvfp4_ms_eden(MsEdenQuantArgs {
            stream: args.stream,
            x: args.x,
            out_fp4: args.out_fp4,
            out_scales: args.out_scales,
            out_global_scales: args.out_global_scales,
            out_chunk_amax: args.out_chunk_amax,
            row_count: args.row_count,
            src_row_len: args.src_row_len,
            dst_row_len: args.dst_row_len,
            global_scale,
            scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
            sign_seed: args.sign_seed,
            scale_seed: args.scale_seed,
        })?;
        Ok(global_scale)
    }
}
