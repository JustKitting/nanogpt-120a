use crate::sweep::{
    analysis::{self, SweepAnalysis},
    config::SweepConfig,
};

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

    normalized_budget(target, weights)
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
    let confidence = if beliefs.is_empty() {
        0.0
    } else {
        beliefs.iter().map(|belief| belief.confidence).sum::<f64>() / beliefs.len() as f64
    }
    .clamp(0.0, 1.0);
    let variance = if beliefs.is_empty() {
        1.0
    } else {
        beliefs.iter().map(|belief| belief.variance).sum::<f64>() / beliefs.len() as f64
    }
    .clamp(0.0, 1.0);
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

fn normalized_budget(target: usize, weights: SourceWeights) -> SourceBudget {
    let raw = [
        weights.guided,
        weights.local,
        weights.factorial,
        weights.variance,
        weights.coverage,
        weights.random,
    ];
    let total = raw.iter().sum::<f64>();
    if total <= 0.0 {
        return SourceBudget {
            random: target,
            ..SourceBudget::default()
        };
    }

    let mut counts = [0usize; 6];
    let mut remainders = [(0usize, 0.0); 6];
    for (index, weight) in raw.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        let exact = target as f64 * *weight / total;
        counts[index] = exact.floor() as usize;
        if counts[index] == 0 {
            counts[index] = 1;
        }
        remainders[index] = (index, exact - exact.floor());
    }

    while counts.iter().sum::<usize>() > target {
        let Some(index) = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .min_by(|a, b| remainders[a.0].1.total_cmp(&remainders[b.0].1))
            .map(|(index, _)| index)
        else {
            break;
        };
        counts[index] -= 1;
    }
    while counts.iter().sum::<usize>() < target {
        remainders.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (index, _) in remainders {
            if counts.iter().sum::<usize>() >= target {
                break;
            }
            if raw[index] > 0.0 {
                counts[index] += 1;
            }
        }
    }

    SourceBudget {
        guided: counts[0],
        local: counts[1],
        factorial: counts[2],
        variance: counts[3],
        coverage: counts[4],
        random: counts[5],
    }
}
