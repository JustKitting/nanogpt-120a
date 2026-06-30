use std::collections::HashSet;

use super::super::{
    analysis,
    candidate::Candidate,
    config::SweepConfig,
    history::Trial,
    rng::SweepRng,
    test_fixtures::{basic_candidate as candidate, quality_config, success_trial as trial},
};

#[test]
fn guided_pool_uses_main_effect_direction() {
    let config = config();
    let trials = [
        trial(candidate(4, 4), 5.0),
        trial(candidate(4, 8), 3.0),
        trial(candidate(16, 4), 3.0),
        trial(candidate(16, 8), 1.0),
    ];
    let center = candidate(8, 4);
    let pool = sample_pool(0x1234, &config, &trials, Some(&center));

    assert_eq!(pool[0].source, "guided");
    assert!(pool[0].candidate.batch_size > center.batch_size);
    assert!(pool[0].candidate.batch_size < 32);
    assert_eq!(pool[0].candidate.n_layer, 8);
    assert!(pool.iter().any(|candidate| candidate.source == "factorial"));
    assert!(pool.iter().any(|candidate| candidate.source == "local"));
    assert!(pool.iter().any(|candidate| candidate.source == "variance"));
    assert!(pool.iter().any(|candidate| candidate.source == "coverage"));
    assert!(pool.iter().any(|candidate| candidate.source == "random"));
}

#[test]
fn local_pool_refines_near_center_hyperparameters() {
    let config = config();
    let trials = (0..32)
        .map(|i| trial(wide_candidate(i), 32.0 - i as f64))
        .collect::<Vec<_>>();
    let mut center = candidate(16, 4);
    center.lr_scale = 2.309_529;
    center.adam_lr_scale = 1.626_648;
    center.nextlat_lr_scale = 1.245_083;
    center.warmup_steps = 87;
    center.start_ratio = 0.183_570;
    center.amuse_beta1 = 0.443_495;
    center.amuse_rho = 0.768_398;
    let pool = sample_pool(0x9933, &config, &trials, Some(&center));
    let locals = pool
        .iter()
        .filter(|candidate| candidate.source == "local")
        .collect::<Vec<_>>();

    assert!(!locals.is_empty());
    assert!(locals.iter().any(|local| {
        local.candidate.build_key() == center.build_key()
            && local.candidate.lr_scale != center.lr_scale
            && local.candidate.adam_lr_scale != center.adam_lr_scale
    }));
    assert!(locals.iter().all(|local| local.candidate.batch_size <= 20));
}

#[test]
fn factorial_pool_can_probe_more_than_four_supported_factors() {
    let config = config();
    let trials = (0..24)
        .map(|i| trial(wide_candidate(i), 24.0 - i as f64))
        .collect::<Vec<_>>();
    let center = wide_candidate(0);
    let pool = sample_pool(0x8822, &config, &trials, Some(&center));
    let factorial = pool
        .iter()
        .find(|candidate| candidate.source == "factorial")
        .unwrap();

    assert!(changed_factors(&center, &factorial.candidate) > 4);
}

#[test]
fn source_budget_keeps_guided_off_without_response_model() {
    let config = config();
    let analysis = analysis::analyze(&[], &config);
    let budget = super::source_budget(40, &analysis, &config);

    assert_eq!(budget.guided, 0);
    assert_eq!(budget.local, 0);
    assert!(budget.variance > 0);
    assert!(budget.coverage > 0);
    assert!(budget.random > 0);
    assert_eq!(budget.total(), 40);
}

#[test]
fn source_budget_moves_toward_guided_when_model_matures() {
    let config = config();
    let empty = analysis::analyze(&[], &config);
    let mature_trials = (0..64)
        .map(|i| trial(wide_candidate(i), 64.0 - i as f64))
        .collect::<Vec<_>>();
    let mature = analysis::analyze(&mature_trials, &config);

    let empty_budget = super::source_budget(40, &empty, &config);
    let mature_budget = super::source_budget(40, &mature, &config);

    assert!(mature_budget.guided > empty_budget.guided);
    assert!(mature_budget.local > empty_budget.local);
    assert!(mature_budget.guided >= mature_budget.coverage);
}

fn changed_factors(left: &Candidate, right: &Candidate) -> usize {
    usize::from(left.batch_size != right.batch_size)
        + usize::from(left.n_layer != right.n_layer)
        + usize::from(left.n_embd != right.n_embd)
        + usize::from(left.aurora_phases != right.aurora_phases)
        + usize::from(left.aurora_blocks != right.aurora_blocks)
        + usize::from(left.lr_scale != right.lr_scale)
        + usize::from(left.adam_lr_scale != right.adam_lr_scale)
        + usize::from(left.nextlat_lr_scale != right.nextlat_lr_scale)
        + usize::from(left.warmup_steps != right.warmup_steps)
        + usize::from(left.start_ratio != right.start_ratio)
        + usize::from(left.amuse_beta1 != right.amuse_beta1)
        + usize::from(left.amuse_rho != right.amuse_rho)
}

fn sample_pool(
    seed: u64,
    config: &SweepConfig,
    trials: &[Trial],
    center: Option<&Candidate>,
) -> Vec<super::PooledCandidate> {
    let analysis = analysis::analyze(trials, config);
    let observed = trials
        .iter()
        .map(|trial| trial.candidate.clone())
        .collect::<Vec<_>>();
    super::sample(
        &HashSet::new(),
        &mut SweepRng::new(seed),
        config,
        &analysis,
        center,
        &observed,
    )
}

fn wide_candidate(i: usize) -> Candidate {
    Candidate {
        batch_size: [4, 8, 12, 16, 20, 24, 28, 32][i % 8],
        n_layer: [4, 8][(i / 3) % 2],
        n_embd: [1024, 2048][(i / 5) % 2],
        n_head: 16,
        aurora_phases: [4, 8, 16][(i / 7) % 3],
        aurora_blocks: [80, 90, 120, 160, 180][(i / 11) % 5],
        lr_scale: 0.5 + (i % 11) as f64 * 0.18,
        adam_lr_scale: 0.5 + (i % 13) as f64 * 0.15,
        nextlat_lr_scale: 0.5 + (i % 17) as f64 * 0.12,
        warmup_steps: 5 + (i * 7) % 96,
        start_ratio: (i % 9) as f64 * 0.025,
        amuse_beta1: 0.2 + (i % 7) as f64 * 0.06,
        amuse_rho: 0.5 + (i % 6) as f64 * 0.08,
    }
}

fn config() -> super::super::config::SweepConfig {
    quality_config(8)
}
