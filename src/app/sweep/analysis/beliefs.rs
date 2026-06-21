use std::collections::BTreeMap;

use super::{super::config::SweepConfig, SweepAnalysis};

#[derive(Clone, Debug)]
pub struct FactorBelief {
    pub factor: String,
    pub direction: f64,
    pub confidence: f64,
    pub variance: f64,
    pub positive_probability: f64,
    pub evidence: usize,
}

#[derive(Default)]
struct Accum {
    direction: f64,
    confidence: f64,
    variance: f64,
    positive_probability: f64,
    evidence: usize,
}

pub fn factor_beliefs(analysis: &SweepAnalysis, config: &SweepConfig) -> Vec<FactorBelief> {
    let mut factors = BTreeMap::<String, Accum>::new();
    for response in &analysis.models {
        let direction_weight = direction_weight(response.name, config);
        let uncertainty_weight = uncertainty_weight(response.name, config);
        if direction_weight == 0.0 && uncertainty_weight == 0.0 {
            continue;
        }
        for effect in response
            .model
            .effects
            .iter()
            .filter(|effect| !effect.name.contains('*'))
        {
            let confidence = directional_confidence(effect.p_positive);
            let weighted = effect.coefficient * direction_weight * confidence;
            let entry = factors.entry(effect.name.clone()).or_default();
            entry.direction += weighted;
            entry.confidence += confidence.abs() * uncertainty_weight.abs();
            entry.variance += effect.stderr * effect.stderr * uncertainty_weight.abs();
            entry.positive_probability += effect.p_positive;
            entry.evidence += 1;
        }
    }
    let mut beliefs = factors
        .into_iter()
        .map(|(factor, value)| FactorBelief {
            factor,
            direction: value.direction,
            confidence: average(value.confidence, value.evidence),
            variance: average(value.variance, value.evidence),
            positive_probability: average(value.positive_probability, value.evidence),
            evidence: value.evidence,
        })
        .collect::<Vec<_>>();
    beliefs.sort_by(|a, b| b.direction.abs().total_cmp(&a.direction.abs()));
    beliefs
}

pub fn tsv(analysis: &SweepAnalysis, config: &SweepConfig) -> String {
    let mut text =
        String::from("factor\tdirection\tconfidence\tvariance\tpositive_probability\tevidence\n");
    for belief in factor_beliefs(analysis, config) {
        text.push_str(&format!(
            "{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\n",
            belief.factor,
            belief.direction,
            belief.confidence,
            belief.variance,
            belief.positive_probability,
            belief.evidence
        ));
    }
    text
}

fn directional_confidence(p_positive: f64) -> f64 {
    (p_positive - 0.5).abs() * 2.0
}

fn average(value: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        value / count as f64
    }
}

fn direction_weight(name: &str, config: &SweepConfig) -> f64 {
    if name.contains("quality") {
        config.sweep_quality_weight
    } else {
        0.0
    }
}

fn uncertainty_weight(name: &str, config: &SweepConfig) -> f64 {
    if name.contains("quality") {
        config.sweep_quality_weight
    } else if name == "stability" {
        config.sweep_stability_weight
    } else {
        0.0
    }
}
