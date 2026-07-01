use cuda_core::DriverError;

use super::super::args::{
    MsEdenDeviceScaleQuantArgs, MsEdenPairDeviceScaleQuantArgs, MsEdenTransposeDeviceScaleQuantArgs,
};
use super::super::launcher::Nvfp4QuantModule;

impl Nvfp4QuantModule {
    pub(super) fn launch_pair_fallback_no_chunk_amax(
        &self,
        args: &mut MsEdenPairDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.fp32_to_nvfp4_ms_eden_device_scale_no_chunk_amax(MsEdenDeviceScaleQuantArgs {
            stream: args.stream,
            x: args.x,
            out_fp4: &mut *args.out_fp4,
            out_scales: &mut *args.out_scales,
            out_global_scales: &mut *args.out_global_scales,
            out_chunk_amax: &mut *args.out_chunk_amax,
            global_scale: &*args.out_global_scale,
            row_count: args.row_count,
            src_row_len: args.src_row_len,
            dst_row_len: args.dst_row_len,
            scale_override: args.scale_override,
            sign_seed: args.sign_seed,
            scale_seed: args.scale_seed,
        })?;

        self.fp32_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax(
            MsEdenTransposeDeviceScaleQuantArgs {
                stream: args.stream,
                x: args.x,
                out_fp4: &mut *args.transpose_out_fp4,
                out_scales: &mut *args.transpose_out_scales,
                out_global_scales: &mut *args.transpose_out_global_scales,
                out_chunk_amax: &mut *args.out_chunk_amax,
                global_scale: &*args.out_global_scale,
                source_rows: args.row_count,
                source_cols: args.src_row_len,
                dst_row_len: args.transpose_dst_row_len,
                scale_override: args.scale_override,
                sign_seed: args.sign_seed,
                scale_seed: args.transpose_scale_seed,
            },
        )
    }
}
