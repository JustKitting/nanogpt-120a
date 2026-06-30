use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

use super::args::{
    MsEdenDeviceScaleQuantArgs, MsEdenPairDeviceScaleQuantArgs, MsEdenQuantArgs,
    MsEdenTransposeDeviceScaleQuantArgs, Nvfp4QuantArgs, Nvfp4QuantRowwiseArgs,
    Nvfp4TransposeMsEdenDeviceScaleQuantArgs, QuartetBackwardMsEdenDeviceScaleQuantArgs,
    QuartetBackwardMsEdenQuantArgs, RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs,
};
use super::kernels;
use super::shape::{
    Fp32PairNoPad, MsEdenPackGrid, RowwiseTransposeNoPad, four_six_grid_config,
    four_six_rowwise_pow2, grid_config,
};
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

const SCALE_OVERRIDE: f32 = 1.0;

pub struct Nvfp4QuantModule {
    pub(super) row_amax: kernels::row_amax::module::LoadedModule,
    four_six: kernels::four_six::module::LoadedModule,
    pub(super) ms_eden: kernels::ms_eden::module::LoadedModule,
}

impl Nvfp4QuantModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            row_amax: kernels::row_amax::module::from_module(module.clone())?,
            four_six: kernels::four_six::module::from_module(module.clone())?,
            ms_eden: kernels::ms_eden::module::from_module(module)?,
        })
    }

    pub fn fp32_to_nvfp4_four_six(&self, args: Nvfp4QuantArgs<'_, '_>) -> Result<(), DriverError> {
        self.launch_fp32_to_nvfp4_four_six(Nvfp4QuantRowwiseArgs {
            stream: args.stream,
            x: args.x,
            amax: args.amax,
            out_fp4: args.out_fp4,
            out_scales: args.out_scales,
            out_global_scale: args.out_global_scale,
            group_count: args.group_count,
            row_len: 0,
        })
    }

    pub fn fp32_to_nvfp4_four_six_rowwise(
        &self,
        args: Nvfp4QuantRowwiseArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        if four_six_rowwise_pow2(args.row_len, args.group_count) {
            return self.launch_fp32_to_nvfp4_four_six_rowwise_pow2(args);
        }

        self.launch_fp32_to_nvfp4_four_six(args)
    }

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
                        .ms_eden
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
                    .ms_eden
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
                .ms_eden
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

    pub fn rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
        &self,
        mut args: RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_rowwise_nvfp4_transpose_global_scale(&mut args)?;
        let pack = MsEdenPackGrid::for_elements(args.source_cols * args.dst_row_len);
        self.ms_eden
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
                        .ms_eden
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
                    .ms_eden
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
                .ms_eden
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

        self.ms_eden
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

    pub fn nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
        &self,
        mut args: Nvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_nvfp4_transpose_global_scale(&mut args)?;
        let pack = MsEdenPackGrid::for_elements(args.source_cols * args.dst_row_len);
        self.ms_eden
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
                .ms_eden
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

        self.ms_eden
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

    fn launch_fp32_to_nvfp4_four_six(
        &self,
        args: Nvfp4QuantRowwiseArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.four_six.fp32_to_nvfp4_four_six_kernel(
            args.stream,
            four_six_grid_config(args.group_count),
            args.x,
            args.amax,
            args.out_fp4,
            args.out_scales,
            args.out_global_scale,
            args.row_len,
            SCALE_OVERRIDE,
        )
    }

    fn launch_fp32_to_nvfp4_four_six_rowwise_pow2(
        &self,
        args: Nvfp4QuantRowwiseArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.four_six.fp32_to_nvfp4_four_six_rowwise_pow2_kernel(
            args.stream,
            four_six_grid_config(args.group_count),
            args.x,
            args.amax,
            args.out_fp4,
            args.out_scales,
            args.out_global_scale,
            args.row_len.trailing_zeros(),
            args.row_len - 1,
            SCALE_OVERRIDE,
        )
    }
}
