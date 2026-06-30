pub(in crate::training::diagnostics) fn changed_bytes(before: &[u8], after: &[u8]) -> usize {
    before
        .iter()
        .zip(after.iter())
        .filter(|(before, after)| before != after)
        .count()
}

pub(in crate::training::diagnostics) struct PendingTensorUpdateDiagnostics {
    pub(super) name: String,
    pub(super) len: usize,
    pub(super) before_bytes: Vec<u8>,
    pub(super) before_scales: Vec<u8>,
    pub(super) before_global: f32,
    pub(super) grad: Vec<f32>,
    pub(super) adam: Option<AdamSnapshot>,
}

pub(super) struct AdamSnapshot {
    pub(super) z_master: Vec<f32>,
    pub(super) x_master: Vec<f32>,
    pub(super) first: Vec<f32>,
    pub(super) second: Vec<f32>,
    pub(super) learning_rate: f32,
    pub(super) weight_decay: f32,
    pub(super) beta1: f32,
    pub(super) beta2: f32,
    pub(super) beta1_correction: f32,
    pub(super) beta2_correction: f32,
    pub(super) eps: f32,
    pub(super) average_coefficient: f32,
}
