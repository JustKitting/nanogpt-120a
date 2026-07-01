use super::super::output::CudaValidOutput;

metric_fields! {
    ValidMetricField, VALID_METRIC_FIELDS, valid_metric_specs, ValidMetricSpec, "" {
        Loss => ("Validation loss", None, false),
        EvalElapsed => ("Eval elapsed", Some("s"), false),
        WindowCount => ("Val windows", None, true),
        CompletedSteps => ("Completed steps", None, true),
    }
}

impl_numeric_metric_spec!(ValidMetricSpec, CudaValidOutput, ValidMetricField::value);

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
