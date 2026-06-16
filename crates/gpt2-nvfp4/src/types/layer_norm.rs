use super::LayerNormTensor;

#[derive(Clone, Debug)]
pub struct LayerNormWeights {
    pub weight: LayerNormTensor,
    pub bias: LayerNormTensor,
}
