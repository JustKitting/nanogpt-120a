use super::{HiddenVectorShape, LayerNormTensor, Nvfp4ShapeInit};

#[derive(Clone, Debug)]
pub struct LayerNormWeights {
    pub weight: LayerNormTensor,
    pub bias: LayerNormTensor,
}

impl LayerNormWeights {
    pub(crate) fn init() -> Self {
        Self {
            weight: HiddenVectorShape::one_tensor(),
            bias: HiddenVectorShape::zero_tensor(),
        }
    }
}
