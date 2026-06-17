mod backward;
mod batch;
mod buffers;
mod forward;
mod grad_block;
mod grads;
mod linear_scratch;
mod operand_scratch;
mod optimizer;
mod optimizer_apply;
mod scratch;
mod tape;
mod tape_block;
mod tape_leaf;

pub use batch::TokenBatch;

use gpt2_nvfp4::{Gpt2, Gpt2Rng};

use crate::AppResult;
use crate::runtime::Runtime;
use crate::upload::UploadedModel;

pub struct Trainer {
    runtime: Runtime,
    model: Gpt2,
    uploaded: UploadedModel,
    buffers: buffers::TrainBuffers,
    rng: Gpt2Rng,
}

pub struct TrainStats {
    pub tokens: usize,
    pub logits: usize,
    pub finite: bool,
    pub nonzero: bool,
    pub loss: f32,
}

impl Trainer {
    pub fn new(seed: u64) -> AppResult<Self> {
        let runtime = Runtime::new()?;
        let stream = runtime.stream.as_ref();
        let mut model = Gpt2::new();
        model.init(seed);
        let weights = model.weights().expect("Gpt2::init must create weights");

        Ok(Self {
            uploaded: UploadedModel::new(stream, weights)?,
            buffers: buffers::TrainBuffers::new(stream)?,
            runtime,
            model,
            rng: Gpt2Rng::new(seed ^ 0xa047_0a91),
        })
    }

    pub fn batch_from_text(&self, text: &str) -> AppResult<TokenBatch> {
        TokenBatch::from_text(self.runtime.stream.as_ref(), text)
    }
}
