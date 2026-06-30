use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::launch::grid_x_config;
use crate::nvfp4_quant::{
    MsEdenDeviceScaleQuantArgs, MsEdenPairDeviceScaleQuantArgs, Nvfp4QuantModule,
    Nvfp4TransposeMsEdenDeviceScaleQuantArgs, QuartetBackwardMsEdenDeviceScaleQuantArgs,
    RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs,
};
use crate::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

use super::{
    LINEAR_BIAS_THREADS_PER_BLOCK, LinearBackwardDeviceScaleArgs, LinearBackwardInputTranspose,
    LinearBackwardModule, LinearBackwardMsEdenArgs, LinearBackwardWeightTranspose,
    MsEdenOperandScratch, bias,
};

impl LinearBackwardModule {
    pub fn backward_ms_eden(
        &self,
        args: LinearBackwardMsEdenArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        if let Some(dbias) = args.dbias {
            self.module.linear_bias_grad_kernel(
                args.stream,
                grid_x_config(
                    bias::grid_dim(args.output_dim),
                    LINEAR_BIAS_THREADS_PER_BLOCK,
                ),
                args.e,
                dbias,
                args.token_count,
                args.output_dim,
            )?;
        }

        let mut scratch = args.scratch;
        let output_k = nvfp4_tc_matmul_padded_k(args.output_dim);
        let token_k = nvfp4_tc_matmul_padded_k(args.token_count);

        args.quant_module
            .fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
                MsEdenPairDeviceScaleQuantArgs {
                    stream: args.stream,
                    x: args.e,
                    out_fp4: &mut *scratch.e_h.bytes,
                    out_scales: &mut *scratch.e_h.scales,
                    out_global_scales: &mut *scratch.e_h.global_scales,
                    transpose_out_fp4: &mut *scratch.e_t_h.bytes,
                    transpose_out_scales: &mut *scratch.e_t_h.scales,
                    transpose_out_global_scales: &mut *scratch.e_t_h.global_scales,
                    out_chunk_amax: &mut *scratch.e_h.chunk_amax,
                    out_global_scale: &mut *scratch.e_h.global_scale,
                    row_count: args.token_count,
                    src_row_len: args.output_dim,
                    dst_row_len: output_k,
                    transpose_dst_row_len: token_k,
                    scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
                    sign_seed: args.sign_seed,
                    scale_seed: args.scale_seed,
                    transpose_scale_seed: args.scale_seed ^ 0x85eb_ca6b,
                    precomputed_chunk_count: args.precomputed_e_amax_chunks,
                },
            )?;
        match args.weight_t {
            LinearBackwardWeightTranspose::Fp32(weight_t) => {
                quantize_operand(
                    args.quant_module,
                    QuantizeOperandArgs {
                        stream: args.stream,
                        x: weight_t,
                        operand: &mut scratch.weight_t_h,
                        row_count: args.input_dim,
                        src_row_len: args.output_dim,
                        dst_row_len: output_k,
                        sign_seed: args.sign_seed,
                        scale_seed: args.scale_seed ^ 0x9e37_79b9,
                        precomputed_chunk_count: None,
                    },
                )?;
            }
            LinearBackwardWeightTranspose::Nvfp4(weight) => {
                args.quant_module
                    .nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
                        Nvfp4TransposeMsEdenDeviceScaleQuantArgs {
                            stream: args.stream,
                            input: weight,
                            out_fp4: &mut *scratch.weight_t_h.bytes,
                            out_scales: &mut *scratch.weight_t_h.scales,
                            out_global_scales: &mut *scratch.weight_t_h.global_scales,
                            out_chunk_amax: &mut *scratch.weight_t_h.chunk_amax,
                            out_global_scale: &mut *scratch.weight_t_h.global_scale,
                            source_rows: args.output_dim,
                            source_cols: args.input_dim,
                            dst_row_len: output_k,
                            sign_seed: args.sign_seed,
                            scale_seed: args.scale_seed ^ 0x9e37_79b9,
                        },
                    )?;
            }
        }
        match args.input_t {
            LinearBackwardInputTranspose::Fp32(input_t) => {
                quantize_operand(
                    args.quant_module,
                    QuantizeOperandArgs {
                        stream: args.stream,
                        x: input_t,
                        operand: &mut scratch.input_t_h,
                        row_count: args.input_dim,
                        src_row_len: args.token_count,
                        dst_row_len: token_k,
                        sign_seed: args.sign_seed,
                        scale_seed: args.scale_seed ^ 0xc2b2_ae35,
                        precomputed_chunk_count: None,
                    },
                )?;
            }
            LinearBackwardInputTranspose::RowwiseNvfp4(input_t) => {
                args.quant_module
                    .rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
                        RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs {
                            stream: args.stream,
                            input: input_t,
                            out_fp4: &mut *scratch.input_t_h.bytes,
                            out_scales: &mut *scratch.input_t_h.scales,
                            out_global_scales: &mut *scratch.input_t_h.global_scales,
                            out_chunk_amax: &mut *scratch.input_t_h.chunk_amax,
                            out_global_scale: &mut *scratch.input_t_h.global_scale,
                            source_rows: args.token_count,
                            source_cols: args.input_dim,
                            dst_row_len: token_k,
                            sign_seed: args.sign_seed,
                            scale_seed: args.scale_seed ^ 0xc2b2_ae35,
                        },
                    )?;
            }
        }

        self.backward_device_scale_cta(LinearBackwardDeviceScaleArgs {
            stream: args.stream,
            e_h: scratch.e_h.rowwise(),
            weight_t_h: scratch.weight_t_h.device_scale_mma_weight(),
            e_t_h: scratch.e_t_h.rowwise(),
            input_t_h: scratch.input_t_h.device_scale_mma_weight(),
            dinput: args.dinput,
            dweight: args.dweight,
            token_count: args.token_count,
            input_dim: args.input_dim,
            output_dim: args.output_dim,
        })
    }
}

struct QuantizeOperandArgs<'a, 'operand, 'scratch> {
    stream: &'a CudaStream,
    x: &'a DeviceBuffer<f32>,
    operand: &'operand mut MsEdenOperandScratch<'scratch>,
    row_count: u32,
    src_row_len: u32,
    dst_row_len: u32,
    sign_seed: u32,
    scale_seed: u32,
    precomputed_chunk_count: Option<u32>,
}

fn quantize_operand(
    module: &Nvfp4QuantModule,
    args: QuantizeOperandArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    let operand = args.operand;
    if let Some(chunk_count) = args.precomputed_chunk_count {
        module.quartet_backward_ms_eden_global_scale_from_chunks(
            args.stream,
            &*operand.chunk_amax,
            &mut *operand.global_scale,
            chunk_count,
        )?;

        return module.fp32_to_nvfp4_ms_eden_device_scale_no_chunk_amax(
            MsEdenDeviceScaleQuantArgs {
                stream: args.stream,
                x: args.x,
                out_fp4: operand.bytes,
                out_scales: operand.scales,
                out_global_scales: operand.global_scales,
                out_chunk_amax: operand.chunk_amax,
                global_scale: &*operand.global_scale,
                row_count: args.row_count,
                src_row_len: args.src_row_len,
                dst_row_len: args.dst_row_len,
                scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
                sign_seed: args.sign_seed,
                scale_seed: args.scale_seed,
            },
        );
    }

    module.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        QuartetBackwardMsEdenDeviceScaleQuantArgs {
            stream: args.stream,
            x: args.x,
            out_fp4: operand.bytes,
            out_scales: operand.scales,
            out_global_scales: operand.global_scales,
            out_chunk_amax: operand.chunk_amax,
            out_global_scale: operand.global_scale,
            row_count: args.row_count,
            src_row_len: args.src_row_len,
            dst_row_len: args.dst_row_len,
            sign_seed: args.sign_seed,
            scale_seed: args.scale_seed,
        },
    )
}
