use super::super::device::GramCorrectionMode as Mode;

pub(in super::super) fn gram_correction_modes() -> [(&'static str, Mode); 18] {
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
