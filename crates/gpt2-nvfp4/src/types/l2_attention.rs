use crate::random::InitRng;
use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::attention::{AttentionModule, QkvProjectionArgs};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantModule, Nvfp4QuantRowwiseArgs};

use super::{HiddenStateDevice, QkvLinear, ResidualLinear};

pub struct AttentionInputNvfp4<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scales: &'a mut DeviceBuffer<f32>,
}

impl<'a> AttentionInputNvfp4<'a> {
    pub fn reborrow(&mut self) -> AttentionInputNvfp4<'_> {
        AttentionInputNvfp4 {
            bytes: &mut *self.bytes,
            scales: &mut *self.scales,
            global_scales: &mut *self.global_scales,
        }
    }
}

pub struct AttentionForwardArgs<'a, 'scratch> {
    pub module: &'a AttentionModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub input_nvfp4: AttentionInputNvfp4<'scratch>,
    pub qkv_weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub qkv_bias: Nvfp4DeviceTensor<'a>,
    pub qkv: &'scratch mut DeviceBuffer<f32>,
    pub hidden: HiddenStateDevice<'a>,
}

#[derive(Clone, Debug)]
pub struct AttentionWeights {
    pub c_attn: QkvLinear,
    pub c_proj: ResidualLinear,
}

impl AttentionWeights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        Self {
            c_attn: QkvLinear::init(rng),
            c_proj: ResidualLinear::init(rng),
        }
    }

    pub fn input_from_embeddings<'a, 'scratch>(
        module: &'a AttentionModule,
        quant_module: &'a Nvfp4QuantModule,
        input_nvfp4: AttentionInputNvfp4<'scratch>,
        qkv_weight: Nvfp4FourSixMmaWeightTensor<'a>,
        qkv_bias: Nvfp4DeviceTensor<'a>,
        qkv: &'scratch mut DeviceBuffer<f32>,
        hidden: HiddenStateDevice<'a>,
    ) -> AttentionForwardArgs<'a, 'scratch> {
        AttentionForwardArgs {
            module,
            quant_module,
            input_nvfp4,
            qkv_weight,
            qkv_bias,
            qkv,
            hidden,
        }
    }

    pub fn forward<'a, 'scratch>(
        &self,
        args: AttentionForwardArgs<'a, 'scratch>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
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
                out_fp4: args.input_nvfp4.bytes,
                out_scales: args.input_nvfp4.scales,
                out_global_scale: args.input_nvfp4.global_scales,
                group_count: (crate::HiddenState::LEN / 16) as u32,
                row_len: crate::GPT2_N_EMBD as u32,
                scale_override: 1.0,
            })?;

        args.module.qkv_projection(QkvProjectionArgs {
            stream,
            input: Nvfp4RowwiseDeviceTensor {
                bytes: &*args.input_nvfp4.bytes,
                scales: &*args.input_nvfp4.scales,
                global_scales: &*args.input_nvfp4.global_scales,
            },
            weight: args.qkv_weight,
            bias: args.qkv_bias,
            out: args.qkv,
            token_count: crate::GPT2_CONTEXT_LEN as u32,
            input_dim: crate::GPT2_N_EMBD as u32,
            output_dim: crate::GPT2_QKV as u32,
        })?;

        Ok(HiddenStateDevice {
            stream,
            residual,
            normalized,
            normalized_amax,
        })
    }
}
