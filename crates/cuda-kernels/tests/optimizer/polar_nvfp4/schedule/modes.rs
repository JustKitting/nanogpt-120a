use super::super::MAX_ITERATIONS;
use super::super::device::GramCorrectionMode;

const TUNED_SCHEDULE: [f32; MAX_ITERATIONS] = [1.0, 1.0, 1.04, 1.01, 1.03, 1.08, 1.0, 1.08];

pub(in super::super) fn production_shape_modes() -> [(&'static str, GramCorrectionMode); 7] {
    [
        ("nvfp4_raw", GramCorrectionMode::Nvfp4GramOnly),
        ("nvfp4_safety105", nvfp4_safety_mode(1.05)),
        ("nvfp4_tuned_schedule", nvfp4_schedule_mode(TUNED_SCHEDULE)),
        ("prefix2_stale_reject3_tuned", stale_reject_schedule_mode(2, 3, TUNED_SCHEDULE)),
        ("prefix3_stale_reject2_tuned", stale_reject_schedule_mode(3, 2, TUNED_SCHEDULE)),
        ("prefix3_stale_reject3_tuned", stale_reject_schedule_mode(3, 3, TUNED_SCHEDULE)),
        ("prefix3_stale_reject5_tuned", stale_reject_schedule_mode(3, 5, TUNED_SCHEDULE)),
    ]
}

const fn nvfp4_safety_mode(coefficient_safety: f32) -> GramCorrectionMode {
    GramCorrectionMode::Nvfp4GramOnlySafety { coefficient_safety }
}

const fn nvfp4_schedule_mode(coefficient_safety: [f32; MAX_ITERATIONS]) -> GramCorrectionMode {
    GramCorrectionMode::Nvfp4GramOnlySchedule { coefficient_safety }
}

const fn stale_reject_schedule_mode(exact_steps: usize, period: usize, coefficient_safety: [f32; MAX_ITERATIONS]) -> GramCorrectionMode {
    GramCorrectionMode::ExactPrefixThenStaleRejectSchedule { exact_steps, period, coefficient_safety }
}
