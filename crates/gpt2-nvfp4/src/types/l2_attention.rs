use crate::random::InitRng;
use cuda_core::DriverError;
use rust_kernels_cuda::attention::{AttentionModule, FakeAttentionArgs};

use super::{HiddenStateDevice, QkvLinear, ResidualLinear};

pub struct AttentionForwardArgs<'a> {
    pub module: &'a AttentionModule,
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

    pub fn input_from_embeddings<'a>(
        module: &'a AttentionModule,
        hidden: HiddenStateDevice<'a>,
    ) -> AttentionForwardArgs<'a> {
        AttentionForwardArgs { module, hidden }
    }

    pub fn forward<'a>(
        &self,
        args: AttentionForwardArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let HiddenStateDevice {
            stream,
            residual,
            normalized,
            normalized_amax,
        } = args.hidden;

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
