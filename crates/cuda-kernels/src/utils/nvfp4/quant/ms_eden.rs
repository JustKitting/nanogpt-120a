use cuda_core::DriverError;

use super::args::{MsEdenDeviceScaleQuantArgs, MsEdenQuantArgs};
use super::launcher::Nvfp4QuantModule;
use super::shape::MsEdenPackGrid;

impl Nvfp4QuantModule {
    pub fn fp32_to_nvfp4_ms_eden(&self, args: MsEdenQuantArgs<'_, '_>) -> Result<(), DriverError> {
        let pack = MsEdenPackGrid::for_elements(args.row_count * args.dst_row_len);
        self.ms_eden_fp32.fp32_to_nvfp4_ms_eden_kernel(
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
        self.ms_eden_fp32.fp32_to_nvfp4_ms_eden_device_scale_kernel(
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
                .ms_eden_fp32
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

        self.ms_eden_fp32
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
}
