macro_rules! metric_fields {
    (
        $field_ty:ident,
        $field_const:ident,
        $spec_ty:ident
        , prefix $prefix:literal
        {
            $($variant:ident => ($name:literal, $unit:expr, $higher_is_better:expr),)+
        }
    ) => {
        metric_fields! {
            @emit $field_ty, $field_const, $spec_ty, $prefix {
                $($variant => ($name, $unit, $higher_is_better),)+
            }
        }
    };
    (
        $field_ty:ident,
        $field_const:ident,
        $spec_ty:ident
        {
            $($variant:ident => ($name:literal, $unit:expr, $higher_is_better:expr),)+
        }
    ) => {
        metric_fields! {
            @emit $field_ty, $field_const, $spec_ty, "" {
                $($variant => ($name, $unit, $higher_is_better),)+
            }
        }
    };
    (
        @emit $field_ty:ident,
        $field_const:ident,
        $spec_ty:ident,
        $prefix:literal
        {
            $($variant:ident => ($name:literal, $unit:expr, $higher_is_better:expr),)+
        }
    ) => {
        #[derive(Clone, Copy)]
        enum $field_ty {
            $($variant,)+
        }

        const $field_const: &[$field_ty] = &[
            $($field_ty::$variant,)+
        ];

        impl $field_ty {
            const fn spec(self) -> $spec_ty {
                match self {
                    $(
                        Self::$variant => $spec_ty {
                            name: concat!($prefix, $name),
                            unit: $unit,
                            higher_is_better: $higher_is_better,
                            field: Self::$variant,
                        },
                    )+
                }
            }
        }
    };
}

mod attention_core_scratch;
mod backward;
mod batch;
mod buffers;
mod data;
mod debug_metrics;
mod diagnostics;
mod eval;
mod forward;
mod generate;
mod grad_block;
mod grad_clip;
mod grads;
mod launch;
mod learning_rate;
mod linear_scratch;
mod next_latent;
mod operand_scratch;
mod optimizer;
mod optimizer_apply;
mod optimizer_aurora;
mod optimizer_state;
mod optimizer_tc_scratch;
pub(crate) mod runtime;
mod save;
mod schedule_free;
mod scratch;
mod tape;
mod tape_block;
mod tape_leaf;
mod update_skip;

pub use batch::{ReusableTokenBatch, TokenBatch};
pub use data::TokenDataLoader;
pub use generate::SamplingConfig;
pub(crate) use launch::launch_from_env;

use gpt2_nvfp4::{GPT2_SEQ_LEN, Gpt2, Gpt2Rng};

use crate::AppResult;
use crate::upload::UploadedModel;
use runtime::Runtime;

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
    pub loss_host_wait_ms: f64,
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
    pub kda_clip_ms: f64,
    pub adam_ms: f64,
    pub adam_lr: f32,
    pub aurora_lr: f32,
    pub grad_norm: f32,
    pub update_skipped: bool,
    pub skip_loss_spike: bool,
    pub skip_grad_norm_spike: bool,
    pub skip_non_finite: bool,
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

    pub fn reusable_default_batch(&self) -> AppResult<ReusableTokenBatch> {
        ReusableTokenBatch::default(self.runtime.stream.as_ref())
    }

    pub fn upload_default_batch<'a>(
        &self,
        batch: &'a mut ReusableTokenBatch,
        tokens: &[u16],
    ) -> AppResult<&'a TokenBatch> {
        batch.upload(self.runtime.stream.as_ref(), tokens)
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
