use super::super::output::CudaTrainOutput;

#[derive(Clone, Copy)]
pub(super) struct TrainMetricSpec {
    name: &'static str,
    unit: Option<&'static str>,
    higher_is_better: bool,
    field: TrainMetricField,
}

metric_fields! {
    TrainMetricField, TRAIN_METRIC_FIELDS, TrainMetricSpec {
        Loss => ("Loss", None, false),
        ForwardMs => ("Forward", Some("ms"), false),
        BackwardEnqueueMs => ("Backward enqueue", Some("ms"), false),
        LossHostWaitMs => ("Loss host wait", Some("ms"), false),
        OptimizerMs => ("Optimizer", Some("ms"), false),
        AuroraMs => ("Aurora", Some("ms"), false),
        KdaClipMs => ("KDA clip", Some("ms"), false),
        AdamMs => ("Adam", Some("ms"), false),
        EmbeddingLookupMs => ("Embedding lookup", Some("ms"), false),
        TokenEmbeddingMs => ("Token embedding", Some("ms"), false),
        FinalNormMs => ("Final norm", Some("ms"), false),
        BlocksMs => ("Blocks", Some("ms"), false),
        GradNorm => ("Grad norm", None, false),
        AdamLr => ("Adam LR", None, false),
        AuroraLr => ("Aurora LR", None, false),
        Tokens => ("Tokens", None, true),
        Logits => ("Logits", None, true),
        Finite => ("Finite", None, true),
        Nonzero => ("Nonzero", None, true),
        UpdateSkipped => ("Update skipped", None, false),
        SkipLossSpike => ("Skip loss spike", None, false),
        SkipGradNormSpike => ("Skip grad norm spike", None, false),
        SkipNonFinite => ("Skip non finite", None, false),
        WindowOffset => ("Window offset", None, true),
        BatchSize => ("Batch size", None, true),
        SeqLen => ("Seq len", None, true),
    }
}

impl TrainMetricSpec {
    pub(super) fn name(self) -> &'static str {
        self.name
    }

    pub(super) fn unit(self) -> Option<&'static str> {
        self.unit
    }

    pub(super) fn higher_is_better(self) -> bool {
        self.higher_is_better
    }

    pub(super) fn value(self, item: &CudaTrainOutput) -> f64 {
        self.field.value(item)
    }
}

pub(super) fn train_metric_specs() -> impl Iterator<Item = TrainMetricSpec> {
    TRAIN_METRIC_FIELDS
        .iter()
        .copied()
        .map(TrainMetricField::spec)
}

impl TrainMetricField {
    fn value(self, item: &CudaTrainOutput) -> f64 {
        let stats = item.stats.as_ref();
        let bool_value = |value: bool| if value { 1.0 } else { 0.0 };
        match self {
            Self::Loss => stats.loss as f64,
            Self::ForwardMs => stats.forward_ms,
            Self::BackwardEnqueueMs => stats.backward_enqueue_ms,
            Self::LossHostWaitMs => stats.loss_host_wait_ms,
            Self::OptimizerMs => stats.optimizer_ms,
            Self::AuroraMs => stats.optimizer.aurora_ms,
            Self::KdaClipMs => stats.optimizer.kda_clip_ms,
            Self::AdamMs => stats.optimizer.adam_ms,
            Self::EmbeddingLookupMs => stats.optimizer.embedding_lookup_ms,
            Self::TokenEmbeddingMs => stats.optimizer.token_embedding_ms,
            Self::FinalNormMs => stats.optimizer.final_norm_ms,
            Self::BlocksMs => stats.optimizer.blocks_ms,
            Self::GradNorm => stats.optimizer.grad_norm as f64,
            Self::AdamLr => stats.optimizer.adam_lr as f64,
            Self::AuroraLr => stats.optimizer.aurora_lr as f64,
            Self::Tokens => stats.tokens as f64,
            Self::Logits => stats.logits as f64,
            Self::Finite => bool_value(stats.finite),
            Self::Nonzero => bool_value(stats.nonzero),
            Self::UpdateSkipped => bool_value(stats.optimizer.update_skipped),
            Self::SkipLossSpike => bool_value(stats.optimizer.skip_loss_spike),
            Self::SkipGradNormSpike => bool_value(stats.optimizer.skip_grad_norm_spike),
            Self::SkipNonFinite => bool_value(stats.optimizer.skip_non_finite),
            Self::WindowOffset => item.window_offset as f64,
            Self::BatchSize => item.batch_size as f64,
            Self::SeqLen => item.seq_len as f64,
        }
    }
}
