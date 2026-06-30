use super::device::GramCorrectionMode as Mode;

pub(super) fn gram_correction_modes() -> [(&'static str, Mode); 18] {
    [
        ("high_precision", Mode::HighPrecision),
        ("nvfp4_gram_only", Mode::Nvfp4GramOnly),
        ("nvfp4_avg2", Mode::Nvfp4GramAverage { samples: 2 }),
        ("nvfp4_avg4", Mode::Nvfp4GramAverage { samples: 4 }),
        ("nvfp4_avg8", Mode::Nvfp4GramAverage { samples: 8 }),
        exact_prefix_nvfp4("prefix2_nvfp4", 2),
        exact_prefix_nvfp4("prefix3_nvfp4", 3),
        exact_prefix_nvfp4_average("prefix2_avg4", 2, 4),
        exact_prefix_nvfp4_average("prefix3_avg4", 3, 4),
        stale("stale_period1", 1),
        stale("stale_period2", 2),
        stale("stale_period3", 3),
        exact_prefix_stale("prefix2_stale2", 2),
        exact_prefix_stale("prefix3_stale2", 3),
        exact_prefix_stale("prefix4_stale2", 4),
        adaptive("adaptive_p2_e03", 2, 3.0e-2),
        adaptive("adaptive_p2_e05", 2, 5.0e-2),
        adaptive("adaptive_p3_e05", 3, 5.0e-2),
    ]
}

pub(super) fn gram_form_correction_modes() -> [(&'static str, Mode); 34] {
    [
        ("high_precision", Mode::HighPrecision),
        high_precision_safety("high_precision_extra_safety101", 1.01),
        high_precision_safety("high_precision_extra_safety103", 1.03),
        high_precision_safety("high_precision_extra_safety104", 1.04),
        high_precision_safety("high_precision_extra_safety1045", 1.045),
        high_precision_safety("high_precision_extra_safety105", 1.05),
        ("nvfp4_gram_only", Mode::Nvfp4GramOnly),
        nvfp4_only_safety("nvfp4_gram_only_extra_safety101", 1.01),
        nvfp4_only_safety("nvfp4_gram_only_extra_safety103", 1.03),
        nvfp4_only_safety("nvfp4_gram_only_extra_safety104", 1.04),
        nvfp4_only_safety("nvfp4_gram_only_extra_safety1045", 1.045),
        nvfp4_only_safety("nvfp4_gram_only_extra_safety105", 1.05),
        late_safety("nvfp4_gram_only_late4_safety1045", 4, 1.045),
        late_safety("nvfp4_gram_only_late4_safety105", 4, 1.05),
        late_safety("nvfp4_gram_only_late3_safety1045", 3, 1.045),
        late_safety("nvfp4_gram_only_late3_safety105", 3, 1.05),
        late_safety("nvfp4_gram_only_late2_safety1045", 2, 1.045),
        late_safety("nvfp4_gram_only_late5_safety105", 5, 1.05),
        ("nvfp4_avg4", Mode::Nvfp4GramAverage { samples: 4 }),
        ("stale_period2", Mode::Stale { period: 2 }),
        ("stale_reject_period2", Mode::StaleReject { period: 2 }),
        stale_reject_safety("stale_reject_p2_extra_safety101", 1.01),
        stale_reject_safety("stale_reject_p2_extra_safety103", 1.03),
        stale_reject_safety("stale_reject_p2_extra_safety105", 1.05),
        stale_scaled("stale_scaled25_period2", 0.25),
        stale_scaled("stale_scaled50_period2", 0.50),
        stale_scaled("stale_scaled75_period2", 0.75),
        exact_prefix_stale_reject("prefix2_stale_reject2", 2),
        exact_prefix_stale_reject("prefix3_stale_reject2", 3),
        exact_prefix_stale_reject_safety("prefix3_stale_reject2_extra_safety101", 1.01),
        exact_prefix_stale_reject_safety("prefix3_stale_reject2_extra_safety103", 1.03),
        exact_prefix_stale_reject_safety("prefix3_stale_reject2_extra_safety105", 1.05),
        exact_prefix_stale_reject_late_safety("prefix3_stale_reject2_late4_safety103", 4),
        exact_prefix_stale_reject_late_safety("prefix3_stale_reject2_late5_safety103", 5),
    ]
}

const fn exact_prefix_nvfp4(name: &'static str, exact_steps: usize) -> (&'static str, Mode) {
    (name, Mode::ExactPrefixThenNvfp4 { exact_steps })
}

const fn exact_prefix_nvfp4_average(
    name: &'static str,
    exact_steps: usize,
    samples: usize,
) -> (&'static str, Mode) {
    (
        name,
        Mode::ExactPrefixThenNvfp4Average {
            exact_steps,
            samples,
        },
    )
}

const fn stale(name: &'static str, period: usize) -> (&'static str, Mode) {
    (name, Mode::Stale { period })
}

const fn exact_prefix_stale(name: &'static str, exact_steps: usize) -> (&'static str, Mode) {
    (
        name,
        Mode::ExactPrefixThenStale {
            exact_steps,
            period: 2,
        },
    )
}

const fn adaptive(
    name: &'static str,
    period: usize,
    max_relative_defect: f32,
) -> (&'static str, Mode) {
    (
        name,
        Mode::Adaptive {
            period,
            max_relative_defect,
        },
    )
}

const fn high_precision_safety(
    name: &'static str,
    coefficient_safety: f32,
) -> (&'static str, Mode) {
    (name, Mode::HighPrecisionSafety { coefficient_safety })
}

const fn nvfp4_only_safety(name: &'static str, coefficient_safety: f32) -> (&'static str, Mode) {
    (name, Mode::Nvfp4GramOnlySafety { coefficient_safety })
}

const fn late_safety(
    name: &'static str,
    start_iter: usize,
    coefficient_safety: f32,
) -> (&'static str, Mode) {
    (
        name,
        Mode::Nvfp4GramOnlyLateSafety {
            start_iter,
            coefficient_safety,
        },
    )
}

const fn stale_scaled(name: &'static str, scale: f32) -> (&'static str, Mode) {
    (name, Mode::StaleScaled { period: 2, scale })
}

const fn stale_reject_safety(name: &'static str, coefficient_safety: f32) -> (&'static str, Mode) {
    (
        name,
        Mode::StaleRejectSafety {
            period: 2,
            coefficient_safety,
        },
    )
}

const fn exact_prefix_stale_reject(name: &'static str, exact_steps: usize) -> (&'static str, Mode) {
    (
        name,
        Mode::ExactPrefixThenStaleReject {
            exact_steps,
            period: 2,
        },
    )
}

const fn exact_prefix_stale_reject_safety(
    name: &'static str,
    coefficient_safety: f32,
) -> (&'static str, Mode) {
    (
        name,
        Mode::ExactPrefixThenStaleRejectSafety {
            exact_steps: 3,
            period: 2,
            coefficient_safety,
        },
    )
}

const fn exact_prefix_stale_reject_late_safety(
    name: &'static str,
    start_iter: usize,
) -> (&'static str, Mode) {
    (
        name,
        Mode::ExactPrefixThenStaleRejectLateSafety {
            exact_steps: 3,
            period: 2,
            start_iter,
            coefficient_safety: 1.03,
        },
    )
}
