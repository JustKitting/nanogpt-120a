use cuda_core::DriverError;

use super::args::Gpt2ForwardArgs;
use super::weights::Gpt2Weights;
use crate::types::{HiddenStateDevice, TokenEmbeddingArgs};
use crate::Gpt2Rng;

#[derive(Clone, Debug, Default)]
pub struct Gpt2 {
    weights: Option<Gpt2Weights>,
}

impl Gpt2 {
    pub const fn new() -> Self {
        Self { weights: None }
    }

    pub fn init(&mut self, seed: u64) {
        let mut rng = Gpt2Rng::new(seed);
        self.init_from_rng(&mut rng);
    }

    pub fn init_from_rng(&mut self, rng: &mut Gpt2Rng) {
        self.weights = Some(Gpt2Weights::init(rng));
    }

    pub fn weights(&self) -> Option<&Gpt2Weights> {
        self.weights.as_ref()
    }

    pub fn weights_mut(&mut self) -> Option<&mut Gpt2Weights> {
        self.weights.as_mut()
    }

    pub fn forward_embeddings<'a>(
        &self,
        args: TokenEmbeddingArgs<'a>,
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
