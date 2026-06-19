use crate::random::InitRng;
use crate::types::{AttentionWeights, LayerNormWeights, MlpWeights};

#[derive(Clone, Debug)]
pub struct Gpt2BlockWeights {
    pub ln_1: LayerNormWeights,
    pub attn: AttentionWeights,
    pub ln_2: LayerNormWeights,
    pub mlp: MlpWeights,
}

impl Gpt2BlockWeights {
    pub(crate) fn init(rng: &mut InitRng, residual_projection_scale: f32) -> Self {
        Self {
            ln_1: LayerNormWeights::init(),
            attn: AttentionWeights::init(rng, residual_projection_scale),
            ln_2: LayerNormWeights::init(),
            mlp: MlpWeights::init(rng, residual_projection_scale),
        }
    }
}
