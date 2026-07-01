use cuda_core::DriverError;

#[path = "ms_eden_pair/exact.rs"]
mod exact;
#[path = "ms_eden_pair/fallback.rs"]
mod fallback;

use super::args::MsEdenPairDeviceScaleQuantArgs;
use super::launcher::Nvfp4QuantModule;
use super::shape::MsEdenPackGrid;

impl Nvfp4QuantModule {
    pub fn fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        &self,
        mut args: MsEdenPairDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let chunk_count = if let Some(chunk_count) = args.precomputed_chunk_count {
            chunk_count
        } else {
            self.tensor_chunk_amax_f32(
                args.stream,
                args.x,
                &mut *args.out_chunk_amax,
                args.row_count * args.src_row_len,
            )?
        };

        self.quartet_backward_ms_eden_global_scale_from_chunks(
            args.stream,
            &*args.out_chunk_amax,
            &mut *args.out_global_scale,
            chunk_count,
        )?;

        let row_pack = MsEdenPackGrid::for_elements(args.row_count * args.dst_row_len);
        let transpose_pack =
            MsEdenPackGrid::for_elements(args.src_row_len * args.transpose_dst_row_len);
        if let Some(result) =
            self.launch_pair_exact_no_chunk_amax(&mut args, row_pack, transpose_pack)
        {
            return result;
        }

        self.launch_pair_fallback_no_chunk_amax(&mut args)
    }
}
