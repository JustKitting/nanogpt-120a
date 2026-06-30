use super::super::super::launch::CudaTrainOutput;
use crate::training::numeric_metric::NumericMetricSpec;

mod value;

#[derive(Clone, Copy)]
pub(super) struct DebugMetricSpec {
    name: &'static str,
    unit: Option<&'static str>,
    higher_is_better: bool,
    field: DebugMetricField,
}

metric_fields! {
    DebugMetricField, DEBUG_METRIC_FIELDS, DebugMetricSpec, prefix "Diagnostic " {
        UpdateCount => ("update count", None, true),
        PositiveUpdateDot => ("positive update dot", None, true),
        ZeroGradChanged => ("zero grad changed", None, false),
        MaxUpdateToWeightRms => ("max update to weight RMS", None, false),
        DlogitsRms => ("dlogits RMS", None, false),
        DlogitsMax => ("dlogits max", None, false),
        DLmHeadRms => ("d lm head RMS", None, false),
        DLmHeadMax => ("d lm head max", None, false),
        DEmbeddingRms => ("d embedding RMS", None, false),
        DEmbeddingMax => ("d embedding max", None, false),
        TokenEmbeddingGlobalBefore => ("token embedding global before", None, false),
        TokenEmbeddingGlobalAfter => ("token embedding global after", None, false),
        TokenEmbeddingChangedBytes => ("token embedding changed bytes", None, true),
        TensorCount => ("tensor count", None, true),
        TensorLenTotal => ("tensor len total", None, true),
        TensorGradRmsMax => ("tensor grad RMS max", None, false),
        TensorGradMaxMax => ("tensor grad max max", None, false),
        TensorGradNonzeroTotal => ("tensor grad nonzero total", None, true),
        TensorGradFiniteAll => ("tensor grad finite all", None, true),
        TensorWeightRmsBeforeMax => ("tensor weight RMS before max", None, false),
        TensorWeightRmsAfterMax => ("tensor weight RMS after max", None, false),
        TensorDeltaRmsMax => ("tensor delta RMS max", None, false),
        TensorDeltaMaxMax => ("tensor delta max max", None, false),
        TensorUpdateToWeightRmsMax => ("tensor update to weight RMS max", None, false),
        TensorDeltaGradDotMaxAbs => ("tensor delta grad dot max abs", None, false),
        TensorDeltaGradCosMaxAbs => ("tensor delta grad cos max abs", None, false),
        TensorPredictedDeltaRmsMax => ("tensor predicted delta RMS max", None, false),
        TensorPredictedDeltaGradDotMaxAbs => ("tensor predicted delta grad dot max abs", None, false),
        TensorPredictedDeltaGradCosMaxAbs => ("tensor predicted delta grad cos max abs", None, false),
        TensorQuantErrorRmsMax => ("tensor quant error RMS max", None, false),
        TensorQuantErrorToPredictedDeltaRmsMax => ("tensor quant error to predicted delta RMS max", None, false),
        TensorChangedBytesTotal => ("tensor changed bytes total", None, true),
        TensorChangedScalesTotal => ("tensor changed scales total", None, true),
        TensorGlobalBeforeMaxAbs => ("tensor global before max abs", None, false),
        TensorGlobalAfterMaxAbs => ("tensor global after max abs", None, false),
    }
}

impl NumericMetricSpec for DebugMetricSpec {
    fn name(self) -> &'static str {
        self.name
    }

    fn unit(self) -> Option<&'static str> {
        self.unit
    }

    fn higher_is_better(self) -> bool {
        self.higher_is_better
    }

    fn value(self, item: &CudaTrainOutput) -> f64 {
        value::debug_metric_value(self.field, item)
    }
}

pub(super) fn debug_metric_specs() -> impl Iterator<Item = DebugMetricSpec> {
    DEBUG_METRIC_FIELDS
        .iter()
        .copied()
        .map(DebugMetricField::spec)
}
