#[derive(Clone, Copy)]
pub enum GramCorrectionMode {
    HighPrecision,
    HighPrecisionSafety {
        coefficient_safety: f32,
    },
    Nvfp4GramOnly,
    Nvfp4GramOnlySafety {
        coefficient_safety: f32,
    },
    Nvfp4GramOnlySchedule {
        coefficient_safety: [f32; 8],
    },
    Nvfp4GramOnlyLateSafety {
        start_iter: usize,
        coefficient_safety: f32,
    },
    Nvfp4GramAverage {
        samples: usize,
    },
    ExactPrefixThenNvfp4 {
        exact_steps: usize,
    },
    ExactPrefixThenNvfp4Average {
        exact_steps: usize,
        samples: usize,
    },
    Stale {
        period: usize,
    },
    StaleReject {
        period: usize,
    },
    StaleRejectSafety {
        period: usize,
        coefficient_safety: f32,
    },
    StaleScaled {
        period: usize,
        scale: f32,
    },
    ExactPrefixThenStale {
        exact_steps: usize,
        period: usize,
    },
    ExactPrefixThenStaleReject {
        exact_steps: usize,
        period: usize,
    },
    ExactPrefixThenStaleRejectSafety {
        exact_steps: usize,
        period: usize,
        coefficient_safety: f32,
    },
    ExactPrefixThenStaleRejectLateSafety {
        exact_steps: usize,
        period: usize,
        start_iter: usize,
        coefficient_safety: f32,
    },
    ExactPrefixThenStaleRejectSchedule {
        exact_steps: usize,
        period: usize,
        coefficient_safety: [f32; 8],
    },
    Adaptive {
        period: usize,
        max_relative_defect: f32,
    },
}

impl GramCorrectionMode {
    pub(super) fn coefficient_safety(self, iter: usize) -> f32 {
        match self {
            GramCorrectionMode::HighPrecisionSafety { coefficient_safety }
            | GramCorrectionMode::Nvfp4GramOnlySafety { coefficient_safety }
            | GramCorrectionMode::StaleRejectSafety {
                coefficient_safety, ..
            }
            | GramCorrectionMode::ExactPrefixThenStaleRejectSafety {
                coefficient_safety, ..
            } => coefficient_safety,
            GramCorrectionMode::Nvfp4GramOnlyLateSafety {
                start_iter,
                coefficient_safety,
            }
            | GramCorrectionMode::ExactPrefixThenStaleRejectLateSafety {
                start_iter,
                coefficient_safety,
                ..
            } => {
                if iter >= start_iter {
                    coefficient_safety
                } else {
                    1.0
                }
            }
            GramCorrectionMode::Nvfp4GramOnlySchedule { coefficient_safety }
            | GramCorrectionMode::ExactPrefixThenStaleRejectSchedule {
                coefficient_safety, ..
            } => coefficient_safety[iter.min(coefficient_safety.len() - 1)],
            _ => 1.0,
        }
    }

    pub(super) fn rejects_stale_steps(self) -> bool {
        matches!(
            self,
            GramCorrectionMode::StaleReject { .. }
                | GramCorrectionMode::StaleRejectSafety { .. }
                | GramCorrectionMode::ExactPrefixThenStaleReject { .. }
                | GramCorrectionMode::ExactPrefixThenStaleRejectSafety { .. }
                | GramCorrectionMode::ExactPrefixThenStaleRejectLateSafety { .. }
                | GramCorrectionMode::ExactPrefixThenStaleRejectSchedule { .. }
        )
    }
}

#[derive(Default)]
pub struct CorrectionStats {
    pub nvfp4_gram_count: usize,
    pub high_precision_gram_count: usize,
    pub max_relative_defect: f32,
    pub last_relative_defect: f32,
    pub rejected_stale_steps: usize,
}
