use std::error::Error;

use super::super::device::{self, GramCorrectionMode};
use super::super::{MAX_ITERATIONS, math};

#[derive(Clone)]
pub(in super::super) struct ScheduleResult {
    pub(in super::super) name: String,
    pub(in super::super) schedule: [f32; MAX_ITERATIONS],
    pub(in super::super) finite: bool,
    pub(in super::super) cosine: f32,
    pub(in super::super) rel_l2: f32,
    pub(in super::super) max_abs: f32,
    pub(in super::super) nvfp4_grams: usize,
    pub(in super::super) hi_grams: usize,
    pub(in super::super) rejected: usize,
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
pub(in super::super) struct ScheduleEval<'a, 'cuda> {
    pub(super) polar: &'a device::Nvfp4Polar<'cuda>,
    pub(super) source: &'a [f32],
    pub(super) expected: &'a [f32],
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) iterations: usize,
}

impl<'a, 'cuda> ScheduleEval<'a, 'cuda> {
    pub(in super::super) fn new(
        polar: &'a device::Nvfp4Polar<'cuda>,
        source: &'a [f32],
        expected: &'a [f32],
        rows: usize,
        cols: usize,
        iterations: usize,
    ) -> Self {
        Self { polar, source, expected, rows, cols, iterations }
    }
}

pub(in super::super) fn evaluate_mode(
    eval: ScheduleEval<'_, '_>,
    name: &str,
    schedule: [f32; MAX_ITERATIONS],
    mode: GramCorrectionMode,
) -> Result<ScheduleResult, Box<dyn Error>> {
    let (actual, stats) = eval.polar.gram_form_corrected_iterations(
        eval.source.to_vec(),
        eval.rows,
        eval.cols,
        eval.iterations,
        mode,
    )?;
    let finite = actual.iter().all(|value| value.is_finite());
    let (cosine, rel_l2, max_abs) = math::finite_error_metrics(&actual, eval.expected, finite);
    Ok(ScheduleResult {
        name: name.to_owned(),
        schedule,
        finite,
        cosine,
        rel_l2,
        max_abs,
        nvfp4_grams: stats.nvfp4_gram_count,
        hi_grams: stats.high_precision_gram_count,
        rejected: stats.rejected_stale_steps,
    })
}
