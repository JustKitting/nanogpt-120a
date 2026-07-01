mod candidates;
mod eval;
mod modes;
mod report;
mod search;

use super::MAX_ITERATIONS;

pub(super) use eval::{ScheduleEval, evaluate_mode};
pub(super) use modes::production_shape_modes;
pub(super) use report::report_schedule_search_best;
pub(super) use search::{search_best_corrected_schedule, search_best_raw_schedule};

fn schedule_result_name(label: &str, name: String) -> String {
    if label.is_empty() {
        name
    } else {
        format!("{label}_{name}")
    }
}

pub(super) fn schedule_name(schedule: &[f32; MAX_ITERATIONS]) -> String {
    schedule
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join("_")
}
