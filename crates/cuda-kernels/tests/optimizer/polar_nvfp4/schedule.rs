use std::error::Error;

use super::{MAX_ITERATIONS, device, math};

#[derive(Clone)]
pub(super) struct ScheduleResult {
    pub(super) name: String,
    pub(super) schedule: [f32; MAX_ITERATIONS],
    pub(super) finite: bool,
    pub(super) cosine: f32,
    pub(super) rel_l2: f32,
    pub(super) max_abs: f32,
    pub(super) nvfp4_grams: usize,
    pub(super) hi_grams: usize,
    pub(super) rejected: usize,
}

impl ScheduleResult {
    pub(super) fn missing() -> Self {
        Self {
            name: "missing".to_owned(),
            schedule: [1.0; MAX_ITERATIONS],
            finite: false,
            cosine: f32::NEG_INFINITY,
            rel_l2: f32::INFINITY,
            max_abs: f32::INFINITY,
            nvfp4_grams: 0,
            hi_grams: 0,
            rejected: 0,
        }
    }

    pub(super) fn is_better_than(&self, other: &Self) -> bool {
        match (self.finite, other.finite) {
            (true, false) => true,
            (false, true) => false,
            (true, true) => self.rel_l2 < other.rel_l2,
            (false, false) => false,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct ScheduleEval<'a, 'cuda> {
    polar: &'a device::Nvfp4Polar<'cuda>,
    source: &'a [f32],
    expected: &'a [f32],
    rows: usize,
    cols: usize,
    iterations: usize,
}

impl<'a, 'cuda> ScheduleEval<'a, 'cuda> {
    pub(super) fn new(
        polar: &'a device::Nvfp4Polar<'cuda>,
        source: &'a [f32],
        expected: &'a [f32],
        rows: usize,
        cols: usize,
        iterations: usize,
    ) -> Self {
        Self {
            polar,
            source,
            expected,
            rows,
            cols,
            iterations,
        }
    }
}

pub(super) fn seed_safety_schedules() -> Vec<[f32; MAX_ITERATIONS]> {
    let mut schedules = Vec::new();
    for value in [1.0, 1.01, 1.02, 1.03, 1.04, 1.045, 1.05, 1.06, 1.08] {
        schedules.push([value; MAX_ITERATIONS]);
    }
    for start in 0..MAX_ITERATIONS {
        for value in [1.03, 1.04, 1.045, 1.05, 1.06] {
            schedules.push(late_safety_schedule(start, value));
        }
    }
    for high in [1.04, 1.045, 1.05, 1.06] {
        let mut schedule = [1.0; MAX_ITERATIONS];
        for (iter, slot) in schedule.iter_mut().enumerate() {
            let t = iter as f32 / (MAX_ITERATIONS - 1) as f32;
            *slot = 1.0 + (high - 1.0) * t;
        }
        schedules.push(schedule);
    }
    schedules.sort_by_key(schedule_name);
    schedules.dedup();
    schedules
}

pub(super) fn corrected_schedule_candidates(
    best_raw: [f32; MAX_ITERATIONS],
) -> Vec<[f32; MAX_ITERATIONS]> {
    let mut schedules = vec![
        best_raw,
        [1.01; MAX_ITERATIONS],
        [1.03; MAX_ITERATIONS],
        [1.05; MAX_ITERATIONS],
    ];
    schedules.extend((2..=5).map(|start| late_safety_schedule(start, 1.03)));
    schedules.sort_by_key(schedule_name);
    schedules.dedup();
    schedules
}

fn late_safety_schedule(start: usize, value: f32) -> [f32; MAX_ITERATIONS] {
    let mut schedule = [1.0; MAX_ITERATIONS];
    schedule[start..].fill(value);
    schedule
}

pub(super) fn search_best_raw_schedule(
    eval: ScheduleEval<'_, '_>,
    label: &str,
    report_progress: bool,
) -> Result<ScheduleResult, Box<dyn Error>> {
    let mut best = ScheduleResult::missing();
    for schedule in seed_safety_schedules() {
        let name = schedule_result_name(label, format!("seed_{}", schedule_name(&schedule)));
        let result = evaluate_schedule(eval, &name, schedule)?;
        if report_progress {
            report_schedule(&result);
        }
        if result.is_better_than(&best) {
            best = result;
        }
    }

    let mut greedy = best.schedule;
    let values = [1.0, 1.01, 1.02, 1.03, 1.04, 1.045, 1.05, 1.06, 1.08];
    for pass in 0..2 {
        for iter in 0..eval.iterations {
            let mut local_best = best.clone();
            for value in values {
                let mut candidate = greedy;
                candidate[iter] = value;
                let name =
                    schedule_result_name(label, format!("greedy_pass{pass}_iter{iter}_{value:.3}"));
                let result = evaluate_schedule(eval, &name, candidate)?;
                if result.is_better_than(&local_best) {
                    local_best = result;
                }
            }
            if local_best.is_better_than(&best) {
                if report_progress {
                    report_schedule(&local_best);
                }
                best = local_best;
                greedy = best.schedule;
            }
        }
    }

    Ok(best)
}

pub(super) fn search_best_corrected_schedule(
    eval: ScheduleEval<'_, '_>,
    raw_schedule: [f32; MAX_ITERATIONS],
    report_progress: bool,
) -> Result<ScheduleResult, Box<dyn Error>> {
    let mut best = ScheduleResult::missing();
    for exact_steps in 1..=3 {
        for period in 2..=5 {
            for schedule in corrected_schedule_candidates(raw_schedule) {
                let result = evaluate_stale_reject_schedule(eval, exact_steps, period, schedule)?;
                if report_progress {
                    report_schedule(&result);
                }
                if result.is_better_than(&best) {
                    best = result;
                }
            }
        }
    }
    Ok(best)
}

fn evaluate_schedule(
    eval: ScheduleEval<'_, '_>,
    name: &str,
    schedule: [f32; MAX_ITERATIONS],
) -> Result<ScheduleResult, Box<dyn Error>> {
    let mode = device::GramCorrectionMode::Nvfp4GramOnlySchedule {
        coefficient_safety: schedule,
    };
    evaluate_mode(eval, name, schedule, mode)
}

fn evaluate_stale_reject_schedule(
    eval: ScheduleEval<'_, '_>,
    exact_steps: usize,
    period: usize,
    schedule: [f32; MAX_ITERATIONS],
) -> Result<ScheduleResult, Box<dyn Error>> {
    let mode = device::GramCorrectionMode::ExactPrefixThenStaleRejectSchedule {
        exact_steps,
        period,
        coefficient_safety: schedule,
    };
    evaluate_mode(
        eval,
        &format!(
            "prefix{exact_steps}_stale_reject{period}_{}",
            schedule_name(&schedule)
        ),
        schedule,
        mode,
    )
}

pub(super) fn evaluate_mode(
    eval: ScheduleEval<'_, '_>,
    name: &str,
    schedule: [f32; MAX_ITERATIONS],
    mode: device::GramCorrectionMode,
) -> Result<ScheduleResult, Box<dyn Error>> {
    let (actual, stats) = eval.polar.gram_form_corrected_iterations(
        eval.source.to_vec(),
        eval.rows,
        eval.cols,
        eval.iterations,
        mode,
    )?;
    let finite = actual.iter().all(|value| value.is_finite());
    Ok(ScheduleResult {
        name: name.to_owned(),
        schedule,
        finite,
        cosine: math::cosine(&actual, eval.expected),
        rel_l2: if finite {
            math::relative_l2(&actual, eval.expected)
        } else {
            f32::INFINITY
        },
        max_abs: if finite {
            math::max_abs_error(&actual, eval.expected)
        } else {
            f32::INFINITY
        },
        nvfp4_grams: stats.nvfp4_gram_count,
        hi_grams: stats.high_precision_gram_count,
        rejected: stats.rejected_stale_steps,
    })
}

pub(super) fn report_schedule(result: &ScheduleResult) {
    report_schedule_result("nvfp4_gram_form_schedule", result);
}

pub(super) fn report_schedule_search_best(kind: &str, result: &ScheduleResult) {
    report_schedule_result(
        format_args!("nvfp4_gram_form_schedule_search_best kind={kind}"),
        result,
    );
}

fn report_schedule_result(label: impl std::fmt::Display, result: &ScheduleResult) {
    println!(
        "{label} name={} finite={} cosine={:.8} rel_l2={:.8e} max_abs={:.8e} nvfp4_grams={} hi_grams={} rejected={} schedule={}",
        result.name,
        result.finite,
        result.cosine,
        result.rel_l2,
        result.max_abs,
        result.nvfp4_grams,
        result.hi_grams,
        result.rejected,
        schedule_name(&result.schedule),
    );
}

fn schedule_result_name(label: &str, name: String) -> String {
    if label.is_empty() {
        name
    } else {
        format!("{label}_{name}")
    }
}

pub(super) fn production_shape_modes() -> [(&'static str, device::GramCorrectionMode); 7] {
    let tuned_schedule = [1.0, 1.0, 1.04, 1.01, 1.03, 1.08, 1.0, 1.08];
    [
        ("nvfp4_raw", device::GramCorrectionMode::Nvfp4GramOnly),
        (
            "nvfp4_safety105",
            device::GramCorrectionMode::Nvfp4GramOnlySafety {
                coefficient_safety: 1.05,
            },
        ),
        (
            "nvfp4_tuned_schedule",
            device::GramCorrectionMode::Nvfp4GramOnlySchedule {
                coefficient_safety: tuned_schedule,
            },
        ),
        (
            "prefix2_stale_reject3_tuned",
            stale_reject_schedule_mode(2, 3, tuned_schedule),
        ),
        (
            "prefix3_stale_reject2_tuned",
            stale_reject_schedule_mode(3, 2, tuned_schedule),
        ),
        (
            "prefix3_stale_reject3_tuned",
            stale_reject_schedule_mode(3, 3, tuned_schedule),
        ),
        (
            "prefix3_stale_reject5_tuned",
            stale_reject_schedule_mode(3, 5, tuned_schedule),
        ),
    ]
}

const fn stale_reject_schedule_mode(
    exact_steps: usize,
    period: usize,
    coefficient_safety: [f32; MAX_ITERATIONS],
) -> device::GramCorrectionMode {
    device::GramCorrectionMode::ExactPrefixThenStaleRejectSchedule {
        exact_steps,
        period,
        coefficient_safety,
    }
}

pub(super) fn schedule_name(schedule: &[f32; MAX_ITERATIONS]) -> String {
    schedule
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join("_")
}
