mod beliefs;
mod design;
mod log_files;
mod logs;
mod regression;
mod report;
mod scoring;
mod stats;

#[cfg(test)]
mod tests;

use super::{config::SweepConfig, history::Trial};

pub use beliefs::factor_beliefs;
pub use regression::Prediction;
pub use scoring::{CandidateScore, score_candidate};

#[derive(Clone, Debug)]
pub struct SweepAnalysis {
    pub models: Vec<ResponseModel>,
    pub trial_count: usize,
    pub stability_prior: Option<BinaryPrior>,
}

#[derive(Clone, Debug)]
pub struct ResponseModel {
    pub name: &'static str,
    pub model: regression::Model,
}

#[derive(Clone, Copy, Debug)]
pub struct BinaryPrior {
    pub n: usize,
    pub positive: f64,
    pub posterior_mean: f64,
}

pub fn analyze(trials: &[Trial], config: &SweepConfig) -> SweepAnalysis {
    let observations = logs::observations(trials);
    let stability_rows = logs::stability_rows(&observations);
    let stability_prior = binary_prior(&stability_rows);
    let models = [
        response_model("screen_quality", logs::screen_quality_rows(&observations, config.screen_max_seconds)),
        response_model("full_quality", logs::full_quality_rows(&observations, config.max_seconds)),
        response_model("stability", stability_rows),
    ]
    .into_iter()
    .flatten()
    .collect();

    SweepAnalysis {
        models,
        trial_count: trials.len(),
        stability_prior,
    }
}

pub fn write(
    sweep_dir: &std::path::Path,
    analysis: &SweepAnalysis,
    config: &SweepConfig,
) -> std::io::Result<()> {
    report::write(sweep_dir, analysis, config)
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

fn response_model(
    name: &'static str,
    rows: Vec<(super::candidate::Candidate, f64)>,
) -> Option<ResponseModel> {
    regression::fit(rows).map(|model| ResponseModel { name, model })
}

fn binary_prior(rows: &[(super::candidate::Candidate, f64)]) -> Option<BinaryPrior> {
    if rows.is_empty() {
        return None;
    }

    let positive = rows
        .iter()
        .map(|(_, value)| value.clamp(0.0, 1.0))
        .sum::<f64>();
    let n = rows.len();
    Some(BinaryPrior {
        n,
        positive,
        posterior_mean: (positive + 1.0) / (n as f64 + 2.0),
    })
}
