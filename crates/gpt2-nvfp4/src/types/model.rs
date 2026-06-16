use cuda_core::DriverError;
use rust_kernels_cuda::attention::AttentionModule;

use crate::random::InitRng;
use crate::{GPT2_N_LAYER, Gpt2Config};

use super::{
    BlockForwardArgs, EmbeddingWeights, Gpt2BlockWeights, HiddenStateDevice, LayerNormWeights,
    TokenPositionEmbeddingArgs,
};

pub struct Gpt2ForwardArgs<'a> {
    pub embeddings: TokenPositionEmbeddingArgs<'a>,
    pub attention_module: &'a AttentionModule,
}

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

    pub fn forward_embeddings<'a>(
        &self,
        args: TokenPositionEmbeddingArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        self.weights()
            .expect("Gpt2::init must be called before forward_embeddings")
            .forward_embeddings(args)
    }

    pub fn forward<'a>(
        &self,
        args: Gpt2ForwardArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        self.weights()
            .expect("Gpt2::init must be called before forward")
            .forward(args)
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
    pub embeddings: EmbeddingWeights,
    pub h: [Gpt2BlockWeights; GPT2_N_LAYER],
    pub ln_f: LayerNormWeights,
}

impl Gpt2Weights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        Self {
            config: Gpt2Config::gpt2_124m(),
            embeddings: EmbeddingWeights::init(rng),
            h: std::array::from_fn(|_| Gpt2BlockWeights::init(rng)),
            ln_f: LayerNormWeights::init(),
        }
    }

    pub fn forward_embeddings<'a>(
        &self,
        args: TokenPositionEmbeddingArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        self.embeddings.forward(args)
    }

    pub fn forward<'a>(
        &self,
        args: Gpt2ForwardArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let mut hidden = self.embeddings.forward(args.embeddings)?;

        for block in &self.h {
            hidden = block.forward(BlockForwardArgs {
                attention_module: args.attention_module,
                hidden,
            })?;
        }

        self.ln_f
            .forward(LayerNormWeights::input_from_block(hidden))
    }
}
