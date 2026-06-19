mod attention_core_scratch;
mod backward;
mod batch;
mod buffers;
mod data;
mod diagnostics;
mod eval;
mod forward;
mod generate;
mod grad_block;
mod grad_clear;
mod grads;
mod learning_rate;
mod linear_scratch;
mod operand_scratch;
mod optimizer;
mod optimizer_apply;
mod optimizer_aurora;
mod optimizer_state;
mod optimizer_tc_scratch;
mod save;
mod schedule_free;
mod scratch;
mod tape;
mod tape_block;
mod tape_leaf;

pub use batch::TokenBatch;
pub use data::TokenDataLoader;
pub use generate::SamplingConfig;

use gpt2_nvfp4::{GPT2_SEQ_LEN, Gpt2, Gpt2Rng};

use crate::AppResult;
use crate::app::runtime::Runtime;
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
    pub forward_ms: f64,
    pub backward_enqueue_ms: f64,
    pub loss_sync_ms: f64,
    pub optimizer_ms: f64,
    pub optimizer: OptimizerTrace,
    pub diagnostics: Option<diagnostics::TrainingDiagnostics>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OptimizerTrace {
    pub embedding_lookup_ms: f64,
    pub token_embedding_ms: f64,
    pub final_norm_ms: f64,
    pub blocks_ms: f64,
    pub aurora_ms: f64,
    pub adam_ms: f64,
    pub adam_lr: f32,
    pub aurora_lr: f32,
}

impl Trainer {
    pub fn new(seed: u64) -> AppResult<Self> {
        let runtime = Runtime::new()?;
        let stream = runtime.stream.as_ref();
        let mut model = Gpt2::new();
        model.init(seed);
        let weights = model.weights().expect("Gpt2::init must create weights");

        let uploaded = UploadedModel::new(stream, weights)?;
        let buffers = buffers::TrainBuffers::new(stream, &runtime, &uploaded)?;

        Ok(Self {
            uploaded,
            buffers,
            runtime,
            model,
            rng: Gpt2Rng::new(seed ^ 0xa047_0a91),
        })
    }

    pub fn batch_from_default_windows(&self, tokens: &[u16]) -> AppResult<TokenBatch> {
        TokenBatch::from_default_batch(self.runtime.stream.as_ref(), tokens)
    }

    pub fn batch_from_windows(&self, tokens: &[u16], batch_size: usize) -> AppResult<TokenBatch> {
        TokenBatch::from_flat_windows(
            self.runtime.stream.as_ref(),
            tokens,
            batch_size,
            GPT2_SEQ_LEN,
        )
    }
}
