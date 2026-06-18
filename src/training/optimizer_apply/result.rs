use super::super::OptimizerTrace;

pub struct WeightUpdateResult {
    pub trace: OptimizerTrace,
    pub diagnostics: Option<super::super::diagnostics::TrainingDiagnostics>,
}
