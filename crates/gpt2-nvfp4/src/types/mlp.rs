use super::{MlpDownLinear, MlpUpLinear};

#[derive(Clone, Debug)]
pub struct MlpWeights {
    pub c_fc: MlpUpLinear,
    pub c_proj: MlpDownLinear,
}
