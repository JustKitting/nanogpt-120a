use super::super::output::CudaValidOutput;

#[derive(Clone, Copy)]
pub(super) struct ValidMetricSpec {
    name: &'static str,
    unit: Option<&'static str>,
    higher_is_better: bool,
    field: ValidMetricField,
}

metric_fields! {
    ValidMetricField, VALID_METRIC_FIELDS, ValidMetricSpec, "" {
        Loss => ("Validation loss", None, false),
        EvalElapsed => ("Eval elapsed", Some("s"), false),
        WindowCount => ("Val windows", None, true),
        CompletedSteps => ("Completed steps", None, true),
    }
}

impl_numeric_metric_spec!(ValidMetricSpec, CudaValidOutput, ValidMetricField::value);

pub(super) fn valid_metric_specs() -> impl Iterator<Item = ValidMetricSpec> {
    VALID_METRIC_FIELDS
        .iter()
        .copied()
        .map(ValidMetricField::spec)
}

impl ValidMetricField {
    fn value(self, item: &CudaValidOutput) -> f64 {
        match self {
            Self::Loss => item.val_loss as f64,
            Self::EvalElapsed => item.eval_elapsed_s,
            Self::WindowCount => item.window_count as f64,
            Self::CompletedSteps => item.completed_steps as f64,
        }
    }
}
