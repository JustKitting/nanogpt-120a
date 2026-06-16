use crate::{GPT2_N_LAYER, Gpt2Config};

use super::{Gpt2BlockWeights, LayerNormWeights, PositionEmbedding, TokenEmbedding};

#[derive(Clone, Debug)]
pub struct Gpt2Weights {
    pub config: Gpt2Config,
    pub wte: TokenEmbedding,
    pub wpe: PositionEmbedding,
    pub h: [Gpt2BlockWeights; GPT2_N_LAYER],
    pub ln_f: LayerNormWeights,
}
