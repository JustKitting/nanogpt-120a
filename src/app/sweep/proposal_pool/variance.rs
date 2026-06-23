use std::collections::HashSet;

use super::super::{
    analysis::{self, SweepAnalysis},
    candidate::Candidate,
    candidate_space,
    config::SweepConfig,
    rng::SweepRng,
};

const HALTON_BASES: [u32; candidate_space::FACTOR_COUNT] =
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

pub fn candidates(
    used: &HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    count: usize,
) -> Vec<Candidate> {
    let mut seen = used.clone();
    let mut ranked = Vec::new();
    let search = (config.candidate_samples.max(1) * 16).max(count * 8);
    let offset = rng.usize(8192);
    push_structured(&mut ranked, &mut seen, offset, search, config, analysis);
    push_random(&mut ranked, &mut seen, rng, count, config, analysis);
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked
        .into_iter()
        .take(count)
        .map(|(candidate, _)| candidate)
        .collect()
}

fn push_structured(
    ranked: &mut Vec<(Candidate, f64)>,
    seen: &mut HashSet<String>,
    offset: usize,
    search: usize,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
) {
    for index in 0..search {
        let candidate = candidate_space::from_unit(halton_units(offset + index + 1));
        if seen.insert(candidate.key()) {
            ranked.push(scored(candidate, config, analysis));
        }
    }
}

fn push_random(
    ranked: &mut Vec<(Candidate, f64)>,
    seen: &mut HashSet<String>,
    rng: &mut SweepRng,
    count: usize,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
) {
    while ranked.len() < count {
        let candidate = unique_random(seen, rng);
        ranked.push(scored(candidate, config, analysis));
    }
}

fn scored(
    candidate: Candidate,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
) -> (Candidate, f64) {
    let score = analysis::score_candidate(analysis, config, &candidate);
    (candidate, score.uncertainty)
}

fn unique_random(seen: &mut HashSet<String>, rng: &mut SweepRng) -> Candidate {
    for _ in 0..4096 {
        let candidate = Candidate::random(rng);
        if seen.insert(candidate.key()) {
            return candidate;
        }
    }
    Candidate::random(rng)
}

fn halton_units(index: usize) -> [f64; candidate_space::FACTOR_COUNT] {
    std::array::from_fn(|dim| radical_inverse(index, HALTON_BASES[dim]))
}

fn radical_inverse(mut index: usize, base: u32) -> f64 {
    let base = base as usize;
    let inv_base = 1.0 / base as f64;
    let mut weight = inv_base;
    let mut value = 0.0;
    while index > 0 {
        value += (index % base) as f64 * weight;
        index /= base;
        weight *= inv_base;
    }
    value
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{candidate_space, halton_units};
    use crate::sweep::{
        analysis, candidate::Candidate, config::SweepConfig, history::Trial, rng::SweepRng,
    };

    #[test]
    fn halton_units_cover_each_factor_range() {
        let rows = (1..=128).map(halton_units).collect::<Vec<_>>();
        for dim in 0..candidate_space::FACTOR_COUNT {
            let min = rows
                .iter()
                .map(|row| row[dim])
                .fold(f64::INFINITY, f64::min);
            let max = rows
                .iter()
                .map(|row| row[dim])
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(min < 0.15, "dim={dim} min={min}");
            assert!(max > 0.85, "dim={dim} max={max}");
        }
    }

    #[test]
    fn variance_candidates_are_unique_structured_points() {
        let config = config();
        let trials = [
            trial(candidate(4, 4), 5.0),
            trial(candidate(4, 8), 4.0),
            trial(candidate(16, 4), 4.0),
            trial(candidate(16, 8), 3.0),
        ];
        let analysis = analysis::analyze(&trials, &config);
        let candidates = super::candidates(
            &HashSet::new(),
            &mut SweepRng::new(0x9911),
            &config,
            &analysis,
            8,
        );
        let unique = candidates
            .iter()
            .map(|candidate| candidate.key())
            .collect::<HashSet<_>>();

        assert_eq!(candidates.len(), 8);
        assert_eq!(unique.len(), candidates.len());
    }

    fn trial(candidate: Candidate, val_loss: f64) -> Trial {
        Trial {
            candidate,
            status: "success".to_string(),
            val_loss: Some(val_loss),
            completed_steps: Some(10),
            elapsed_s: Some(900.0),
            screen_val_loss: Some(val_loss + 1.0),
            screen_completed_steps: Some(10),
            screen_elapsed_s: Some(30.0),
            screen_reason: Some("screen_loss_improved".to_string()),
            log_path: "train.log".into(),
        }
    }

    fn candidate(batch_size: usize, n_layer: usize) -> Candidate {
        Candidate {
            batch_size,
            n_layer,
            n_embd: 1024,
            n_head: 16,
            aurora_phases: 4,
            aurora_blocks: 80,
            lr_scale: 1.0,
            adam_lr_scale: 1.0,
            nextlat_lr_scale: 1.0,
            warmup_steps: 20,
            start_ratio: 0.1,
            amuse_beta1: 0.4,
            amuse_rho: 0.8,
        }
    }

    fn config() -> SweepConfig {
        SweepConfig {
            trials: 4,
            random_trials: 0,
            candidate_samples: 24,
            max_seconds: 900.0,
            screen_max_seconds: 30.0,
            sweep_quality_weight: 1.0,
            sweep_stability_weight: 0.75,
            sweep_exploration_weight: 0.35,
            log_interval: 500,
            dataset: "synth".to_string(),
            arch: "sm_120a".to_string(),
            cuda_device: None,
            sweep_dir: None,
            seed_history: "notes/sweep_seed_current.tsv".into(),
            baseline: "notes/sweep_baseline.env".into(),
            seed: 0x4750_5432,
            dry_run: false,
        }
    }
}
