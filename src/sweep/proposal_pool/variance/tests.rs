use std::collections::HashSet;

use crate::sweep::{
    analysis, candidate_space,
    rng::SweepRng,
    test_fixtures::{basic_candidate as candidate, config as sweep_config, success_trial as trial},
};

use super::halton;

#[test]
fn halton_units_cover_each_factor_range() {
    let rows = (1..=128).map(halton::units).collect::<Vec<_>>();
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
    let config = sweep_config(0, 24);
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
