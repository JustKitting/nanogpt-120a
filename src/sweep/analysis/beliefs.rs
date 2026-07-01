use std::collections::BTreeMap;

use super::{super::config::SweepConfig, regression::Effect, SweepAnalysis};

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

impl Accum {
    fn add(&mut self, effect: &Effect, weights: &ResponseWeights) {
        let confidence = directional_confidence(effect.p_positive); self.direction += effect.coefficient * weights.direction * confidence;
        self.confidence += confidence.abs() * weights.uncertainty.abs();
        self.variance += effect.stderr * effect.stderr * weights.uncertainty.abs();
        self.positive_probability += effect.p_positive; self.evidence += 1;
    }

    fn belief(self, factor: String) -> FactorBelief {
        FactorBelief { factor, direction: self.direction, confidence: average(self.confidence, self.evidence), variance: average(self.variance, self.evidence), positive_probability: average(self.positive_probability, self.evidence), evidence: self.evidence }
    }
}

pub fn factor_beliefs(analysis: &SweepAnalysis, config: &SweepConfig) -> Vec<FactorBelief> {
    let mut factors = BTreeMap::<String, Accum>::new();
    for response in &analysis.models {
        let weights = response_weights(response.name, config);
        if weights.direction == 0.0 && weights.uncertainty == 0.0 {
            continue;
        }
        for effect in response
            .model
            .effects
            .iter()
            .filter(|effect| !effect.name.contains('*'))
        {
            factors.entry(effect.name.clone()).or_default().add(effect, &weights);
        }
    }
    let mut beliefs = factors
        .into_iter()
        .map(|(factor, value)| value.belief(factor))
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

struct ResponseWeights { direction: f64, uncertainty: f64 }

fn response_weights(name: &str, config: &SweepConfig) -> ResponseWeights {
    if name.contains("quality") {
        ResponseWeights { direction: config.sweep_quality_weight, uncertainty: config.sweep_quality_weight }
    } else if name == "stability" {
        ResponseWeights { direction: 0.0, uncertainty: config.sweep_stability_weight }
    } else {
        ResponseWeights { direction: 0.0, uncertainty: 0.0 }
    }
}
