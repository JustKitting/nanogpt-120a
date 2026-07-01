#[derive(Default)]
pub struct CorrectionStats {
    pub nvfp4_gram_count: usize,
    pub high_precision_gram_count: usize,
    pub max_relative_defect: f32,
    pub last_relative_defect: f32,
    pub rejected_stale_steps: usize,
}

impl CorrectionStats {
    pub(super) fn pending() -> Self {
        Self {
            last_relative_defect: f32::INFINITY,
            ..Self::default()
        }
    }
}
