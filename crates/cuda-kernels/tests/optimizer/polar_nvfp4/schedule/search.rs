use std::error::Error;

use super::super::MAX_ITERATIONS;
use super::super::device::GramCorrectionMode;
use super::candidates::{SAFETY_VALUES, corrected_schedule_candidates, seed_safety_schedules};
use super::eval::{ScheduleEval, ScheduleResult, evaluate_mode};
use super::report::report_schedule;
use super::{schedule_name, schedule_result_name};

pub(in super::super) fn search_best_raw_schedule(
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
    for pass in 0..2 {
        for iter in 0..eval.iterations {
            let mut local_best = best.clone();
            for value in SAFETY_VALUES {
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

pub(in super::super) fn search_best_corrected_schedule(
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
    let mode = GramCorrectionMode::Nvfp4GramOnlySchedule { coefficient_safety: schedule };
    evaluate_mode(eval, name, schedule, mode)
}

fn evaluate_stale_reject_schedule(
    eval: ScheduleEval<'_, '_>,
    exact_steps: usize,
    period: usize,
    schedule: [f32; MAX_ITERATIONS],
) -> Result<ScheduleResult, Box<dyn Error>> {
    let mode = GramCorrectionMode::ExactPrefixThenStaleRejectSchedule { exact_steps, period, coefficient_safety: schedule };
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
