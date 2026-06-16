use crate::random::InitRng;
use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::attention::{AttentionModule, FakeAttentionArgs};
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
        hidden: HiddenStateDevice<'a>,
    ) -> AttentionForwardArgs<'a, 'scratch> {
        AttentionForwardArgs {
            module,
            quant_module,
            input_nvfp4,
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

        args.module.fake_attention(FakeAttentionArgs::new(
            stream,
            normalized,
            crate::HiddenState::LEN as u32,
        ))?;

        Ok(HiddenStateDevice {
            stream,
            residual,
            normalized,
            normalized_amax,
        })
    }
}
