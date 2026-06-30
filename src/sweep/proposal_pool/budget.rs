use crate::sweep::{
    analysis::{self, SweepAnalysis},
    config::SweepConfig,
};

mod normalize;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SourceBudget {
    pub(super) guided: usize,
    pub(super) local: usize,
    pub(super) factorial: usize,
    pub(super) variance: usize,
    pub(super) coverage: usize,
    pub(super) random: usize,
}

impl SourceBudget {
    pub(super) fn total(self) -> usize {
        self.guided + self.local + self.factorial + self.variance + self.coverage + self.random
    }
}

pub(super) fn source_budget(
    target: usize,
    analysis: &SweepAnalysis,
    config: &SweepConfig,
) -> SourceBudget {
    let mut weights = source_weights(analysis, config);
    if target < 4 {
        weights.coverage += weights.random;
        weights.random = 0.0;
    }

    normalize::normalized_budget(target, weights)
}

#[derive(Clone, Copy, Debug)]
struct SourceWeights {
    guided: f64,
    local: f64,
    factorial: f64,
    variance: f64,
    coverage: f64,
    random: f64,
}

fn source_weights(analysis: &SweepAnalysis, config: &SweepConfig) -> SourceWeights {
    let beliefs = analysis::factor_beliefs(analysis, config);
    let confidence = mean_or(&beliefs, 0.0, |belief| belief.confidence).clamp(0.0, 1.0);
    let variance = mean_or(&beliefs, 1.0, |belief| belief.variance).clamp(0.0, 1.0);
    let model_maturity =
        (analysis.trial_count as f64 / (analysis.trial_count as f64 + 12.0)).clamp(0.0, 1.0);
    let has_response_model = analysis
        .models
        .iter()
        .any(|model| model.name.contains("quality"));
    let exploitation = if has_response_model {
        (0.25 + 0.55 * model_maturity * confidence).clamp(0.0, 0.8)
    } else {
        0.0
    };
    let uncertainty = (1.0 - confidence).max(variance.sqrt()).clamp(0.0, 1.0);

    SourceWeights {
        guided: exploitation,
        local: if has_response_model {
            (0.2 + 0.3 * model_maturity).clamp(0.0, 0.45)
        } else {
            0.0
        },
        factorial: 0.15 + 0.25 * uncertainty,
        variance: 0.2 + 0.35 * uncertainty,
        coverage: 0.15 + 0.3 * (1.0 - model_maturity),
        random: 0.1 + 0.2 * (1.0 - model_maturity),
    }
}

fn mean_or<T>(items: &[T], empty: f64, value: impl Fn(&T) -> f64) -> f64 {
    if items.is_empty() {
        empty
    } else {
        items.iter().map(value).sum::<f64>() / items.len() as f64
    }
}
