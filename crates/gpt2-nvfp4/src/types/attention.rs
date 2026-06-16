use super::{QkvLinear, ResidualLinear};

#[derive(Clone, Debug)]
pub struct AttentionWeights {
    pub c_attn: QkvLinear,
    pub c_proj: ResidualLinear,
}
