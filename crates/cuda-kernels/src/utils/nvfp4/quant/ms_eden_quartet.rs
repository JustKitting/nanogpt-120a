use cuda_core::DriverError;

use super::args::{
    MsEdenQuantArgs, QuartetBackwardMsEdenDeviceScaleQuantArgs, QuartetBackwardMsEdenQuantArgs,
};
use super::launcher::Nvfp4QuantModule;
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

impl Nvfp4QuantModule {
    pub fn fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale(
        &self,
        mut args: QuartetBackwardMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_fp32_quartet_backward_ms_eden_global_scale(&mut args)?;
        self.fp32_to_nvfp4_ms_eden_device_scale(
            args.device_scale_args(QUARTET_MS_EDEN_SCALE_OVERRIDE),
        )
    }

    pub fn fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        &self,
        mut args: QuartetBackwardMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_fp32_quartet_backward_ms_eden_global_scale(&mut args)?;
        self.fp32_to_nvfp4_ms_eden_device_scale_no_chunk_amax(
            args.device_scale_args(QUARTET_MS_EDEN_SCALE_OVERRIDE),
        )
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
