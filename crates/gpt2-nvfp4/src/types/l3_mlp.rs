use crate::random::InitRng;
use cuda_core::DriverError;

use super::{HiddenStateDevice, MlpDownLinear, MlpUpLinear};

pub struct MlpForwardArgs<'a> {
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

    pub fn input_from_attention<'a>(hidden: HiddenStateDevice<'a>) -> MlpForwardArgs<'a> {
        MlpForwardArgs { hidden }
    }

    pub fn forward<'a>(
        &self,
        args: MlpForwardArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        Ok(args.hidden)
    }
}
