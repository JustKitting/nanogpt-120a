use cuda_core::DriverError;

use super::forward;
use super::tensors::AttentionForwardArgs;
use crate::random::InitRng;
use crate::types::{HiddenStateDevice, QkvLinear, ResidualLinear};

#[derive(Clone, Debug)]
pub struct AttentionWeights {
    pub c_attn: QkvLinear,
    pub c_proj: ResidualLinear,
}

impl AttentionWeights {
    pub(crate) fn init(rng: &mut InitRng, residual_projection_scale: f32) -> Self {
        Self {
            c_attn: QkvLinear::init(rng),
            c_proj: ResidualLinear::init_with_weight_scale(rng, residual_projection_scale),
        }
    }

    pub fn forward<'a, 'scratch>(
        args: AttentionForwardArgs<'a, 'scratch>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        forward::forward(args)
    }
}
