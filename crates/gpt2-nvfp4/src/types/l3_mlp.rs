use crate::random::InitRng;
use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::mlp::{MlpDownResidualArgs, MlpModule, MlpUpRelu2Args};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantModule, Nvfp4QuantRowwiseArgs, RowAmaxArgs};

use super::{HiddenStateDevice, HiddenStateNvfp4, MlpActivationNvfp4, MlpDownLinear, MlpUpLinear};

#[derive(Clone, Copy)]
pub struct MlpUpTensors<'a> {
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
}

#[derive(Clone, Copy)]
pub struct MlpDownTensors<'a> {
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
}

#[derive(Clone, Copy)]
pub struct MlpProjectionTensors<'a> {
    pub up: MlpUpTensors<'a>,
    pub down: MlpDownTensors<'a>,
}

pub struct MlpScratch<'scratch> {
    pub input_nvfp4: HiddenStateNvfp4<'scratch>,
    pub activation_nvfp4: MlpActivationNvfp4<'scratch>,
    pub activation: &'scratch mut DeviceBuffer<f32>,
}

pub struct MlpForwardArgs<'a, 'scratch> {
    pub module: &'a MlpModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub scratch: MlpScratch<'scratch>,
    pub projections: MlpProjectionTensors<'a>,
    pub hidden: HiddenStateDevice<'a>,
}

#[derive(Clone, Debug)]
pub struct MlpWeights {
    pub c_fc: MlpUpLinear,
    pub c_proj: MlpDownLinear,
}

impl MlpWeights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        Self {
            c_fc: MlpUpLinear::init(rng),
            c_proj: MlpDownLinear::init(rng),
        }
    }

    pub fn input_from_attention<'a, 'scratch>(
        module: &'a MlpModule,
        quant_module: &'a Nvfp4QuantModule,
        scratch: MlpScratch<'scratch>,
        projections: MlpProjectionTensors<'a>,
        hidden: HiddenStateDevice<'a>,
    ) -> MlpForwardArgs<'a, 'scratch> {
        MlpForwardArgs {
            module,
            quant_module,
            scratch,
            projections,
            hidden,
        }
    }

    pub fn forward<'a, 'scratch>(
        args: MlpForwardArgs<'a, 'scratch>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let input_nvfp4 = args.scratch.input_nvfp4;
        let activation_nvfp4 = args.scratch.activation_nvfp4;
        let HiddenStateDevice {
            stream,
            residual,
            normalized,
            normalized_amax,
        } = args.hidden;

        args.quant_module
            .fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
                stream,
                x: normalized,
                amax: normalized_amax,
                out_fp4: &mut *input_nvfp4.bytes,
                out_scales: &mut *input_nvfp4.scales,
                out_global_scale: &mut *input_nvfp4.global_scales,
                group_count: (crate::HiddenState::LEN / 16) as u32,
                row_len: crate::GPT2_N_EMBD as u32,
            })?;

        args.module.up_relu2(MlpUpRelu2Args {
            stream,
            input: Nvfp4RowwiseDeviceTensor {
                bytes: &*input_nvfp4.bytes,
                scales: &*input_nvfp4.scales,
                global_scales: &*input_nvfp4.global_scales,
            },
            weight: args.projections.up.weight,
            bias: args.projections.up.bias,
            out: args.scratch.activation,
            token_count: crate::GPT2_CONTEXT_LEN as u32,
            input_dim: crate::GPT2_N_EMBD as u32,
            output_dim: crate::GPT2_MLP as u32,
        })?;

        args.quant_module.row_amax_f32(RowAmaxArgs {
            stream,
            x: args.scratch.activation,
            out: normalized_amax,
            row_count: crate::GPT2_CONTEXT_LEN as u32,
            row_len: crate::GPT2_MLP as u32,
        })?;

        args.quant_module
            .fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
                stream,
                x: args.scratch.activation,
                amax: normalized_amax,
                out_fp4: &mut *activation_nvfp4.bytes,
                out_scales: &mut *activation_nvfp4.scales,
                out_global_scale: &mut *activation_nvfp4.global_scales,
                group_count: (crate::MlpActivation::LEN / 16) as u32,
                row_len: crate::GPT2_MLP as u32,
            })?;

        args.module.down_residual(MlpDownResidualArgs {
            stream,
            input: Nvfp4RowwiseDeviceTensor {
                bytes: &*activation_nvfp4.bytes,
                scales: &*activation_nvfp4.scales,
                global_scales: &*activation_nvfp4.global_scales,
            },
            weight: args.projections.down.weight,
            bias: args.projections.down.bias,
            residual,
            token_count: crate::GPT2_CONTEXT_LEN as u32,
            input_dim: crate::GPT2_MLP as u32,
            output_dim: crate::GPT2_N_EMBD as u32,
        })?;

        Ok(HiddenStateDevice {
            stream,
            residual,
            normalized,
            normalized_amax,
        })
    }
}
