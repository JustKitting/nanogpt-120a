use crate::random::InitRng;
use crate::{GPT2_N_LAYER, Gpt2Config};

use super::{
    Gpt2BlockWeights, LayerNormWeights, Nvfp4ShapeInit, PositionEmbedding, PositionEmbeddingShape,
    TokenEmbedding, TokenEmbeddingShape,
};

#[derive(Clone, Debug)]
pub struct Gpt2 {
    weights: Option<Gpt2Weights>,
}

impl Gpt2 {
    pub const fn new() -> Self {
        Self { weights: None }
    }

    pub fn init(&mut self, seed: u64) {
        let mut rng = InitRng::new(seed);
        self.weights = Some(Gpt2Weights::init(&mut rng));
    }

    pub fn weights(&self) -> Option<&Gpt2Weights> {
        self.weights.as_ref()
    }

    pub fn weights_mut(&mut self) -> Option<&mut Gpt2Weights> {
        self.weights.as_mut()
    }
}

impl Default for Gpt2 {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct Gpt2Weights {
    pub config: Gpt2Config,
    pub wte: TokenEmbedding,
    pub wpe: PositionEmbedding,
    pub h: [Gpt2BlockWeights; GPT2_N_LAYER],
    pub ln_f: LayerNormWeights,
}

impl Gpt2Weights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        Self {
            config: Gpt2Config::gpt2_124m(),
            wte: TokenEmbeddingShape::smooth_tensor(rng),
            wpe: PositionEmbeddingShape::smooth_tensor(rng),
            h: std::array::from_fn(|_| Gpt2BlockWeights::init(rng)),
            ln_f: LayerNormWeights::init(),
        }
    }
}
