use cuda_core::DriverError;

use super::args::{
    MsEdenDeviceScaleQuantArgs, MsEdenPairDeviceScaleQuantArgs, MsEdenTransposeDeviceScaleQuantArgs,
};
use super::launcher::Nvfp4QuantModule;
use super::shape::{Fp32PairNoPad, MsEdenPackGrid, grid_config};

impl Nvfp4QuantModule {
    pub fn fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        &self,
        args: MsEdenPairDeviceScaleQuantArgs<'_, '_>,
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
        if row_pack.is_exact() && transpose_pack.is_exact() {
            let grid = grid_config(row_pack.grid_dim + transpose_pack.grid_dim);
            if let Some(no_pad) = Fp32PairNoPad::new(
                args.row_count,
                args.src_row_len,
                args.dst_row_len,
                args.transpose_dst_row_len,
            ) {
                if let Some(pow2) = no_pad.pow2() {
                    return self
                        .ms_eden_fp32_pair
                        .fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_pow2_kernel(
                            args.stream,
                            grid,
                            args.x,
                            args.out_fp4,
                            args.out_scales,
                            args.out_global_scales,
                            args.transpose_out_fp4,
                            args.transpose_out_scales,
                            args.transpose_out_global_scales,
                            &*args.out_global_scale,
                            row_pack.grid_dim,
                            args.src_row_len,
                            pow2.chunks_per_row_shift,
                            pow2.transpose_chunks_per_row_shift,
                            args.scale_override,
                            args.sign_seed,
                            args.scale_seed,
                            args.transpose_scale_seed,
                        );
                }

                return self
                    .ms_eden_fp32_pair
                    .fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_kernel(
                        args.stream,
                        grid,
                        args.x,
                        args.out_fp4,
                        args.out_scales,
                        args.out_global_scales,
                        args.transpose_out_fp4,
                        args.transpose_out_scales,
                        args.transpose_out_global_scales,
                        &*args.out_global_scale,
                        row_pack.grid_dim,
                        args.src_row_len,
                        no_pad.chunks_per_row,
                        no_pad.transpose_chunks_per_row,
                        args.scale_override,
                        args.sign_seed,
                        args.scale_seed,
                        args.transpose_scale_seed,
                    );
            }

            return self
                .ms_eden_fp32_pair
                .fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
                    args.stream,
                    grid,
                    args.x,
                    args.out_fp4,
                    args.out_scales,
                    args.out_global_scales,
                    args.transpose_out_fp4,
                    args.transpose_out_scales,
                    args.transpose_out_global_scales,
                    &*args.out_global_scale,
                    row_pack.grid_dim,
                    args.row_count,
                    args.src_row_len,
                    args.dst_row_len,
                    args.transpose_dst_row_len,
                    args.scale_override,
                    args.sign_seed,
                    args.scale_seed,
                    args.transpose_scale_seed,
                );
        }

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
            scale_override: args.scale_override,
            sign_seed: args.sign_seed,
            scale_seed: args.scale_seed,
        })?;

        self.fp32_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax(
            MsEdenTransposeDeviceScaleQuantArgs {
                stream: args.stream,
                x: args.x,
                out_fp4: args.transpose_out_fp4,
                out_scales: args.transpose_out_scales,
                out_global_scales: args.transpose_out_global_scales,
                out_chunk_amax: args.out_chunk_amax,
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
