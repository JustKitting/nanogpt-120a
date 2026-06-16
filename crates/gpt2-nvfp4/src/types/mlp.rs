use crate::random::InitRng;

use super::{MlpDownLinear, MlpUpLinear};

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
}
