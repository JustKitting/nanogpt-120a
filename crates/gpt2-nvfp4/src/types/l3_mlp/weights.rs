use cuda_core::DriverError;

use super::forward;
use super::tensors::MlpForwardArgs;
use crate::random::InitRng;
use crate::types::{HiddenStateDevice, MlpDownLinear, MlpUpLinear};

#[derive(Clone, Debug)]
pub struct MlpWeights {
    pub c_fc: MlpUpLinear,
    pub c_proj: MlpDownLinear,
}

impl MlpWeights {
    pub(crate) fn init(rng: &mut InitRng, residual_projection_scale: f32) -> Self {
        Self {
            c_fc: MlpUpLinear::init(rng),
            c_proj: MlpDownLinear::init_with_weight_scale(rng, residual_projection_scale),
        }
    }

    pub fn forward<'a, 'scratch>(
        args: MlpForwardArgs<'a, 'scratch>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        forward::forward(args)
    }
}
