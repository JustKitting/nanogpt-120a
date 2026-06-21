mod design;
mod factors;
mod log_files;
mod logs;
mod regression;
mod report;
mod scoring;
mod stats;

#[cfg(test)]
mod tests;

use super::{config::SweepConfig, history::Trial};

pub use regression::Prediction;
pub use scoring::{CandidateScore, score_candidate};

#[derive(Clone, Debug)]
pub struct SweepAnalysis {
    pub models: Vec<ResponseModel>,
    pub trial_count: usize,
}

#[derive(Clone, Debug)]
pub struct ResponseModel {
    pub name: &'static str,
    pub model: regression::Model,
}

pub fn analyze(trials: &[Trial], config: &SweepConfig) -> SweepAnalysis {
    let observations = logs::observations(trials);
    let mut models = Vec::new();
    push_model(
        &mut models,
        "screen_quality",
        logs::screen_quality_rows(&observations, config.screen_steps),
    );
    push_model(
        &mut models,
        "screen_tokens_per_s",
        logs::screen_speed_rows(&observations),
    );
    push_model(
        &mut models,
        "full_quality",
        logs::full_quality_rows(&observations),
    );
    push_model(
        &mut models,
        "full_tokens_per_s",
        logs::full_speed_rows(&observations),
    );
    push_model(
        &mut models,
        "stability",
        logs::stability_rows(&observations),
    );

    SweepAnalysis {
        models,
        trial_count: trials.len(),
    }
}

pub fn write(sweep_dir: &std::path::Path, analysis: &SweepAnalysis) -> std::io::Result<()> {
    report::write(sweep_dir, analysis)
}

pub fn print_summary(analysis: &SweepAnalysis) {
    for response in &analysis.models {
        if let Some(effect) = response.model.effects.first() {
            println!(
                "sweep_analysis response={} n={} top={} coef={:.4} p_positive={:.3}",
                response.name, response.model.n, effect.name, effect.coefficient, effect.p_positive
            );
        }
    }
}

fn push_model(
    models: &mut Vec<ResponseModel>,
    name: &'static str,
    rows: Vec<(super::candidate::Candidate, f64)>,
) {
    if let Some(model) = regression::fit(rows) {
        models.push(ResponseModel { name, model });
    }
}
