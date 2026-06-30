mod baseline;
mod model;
mod random;

use std::collections::HashSet;

use crate::sweep::{
    analysis,
    candidate::Candidate,
    config::SweepConfig,
    history::Trial,
    optimizer::{self, Proposal},
};

use super::fixtures::rng;

fn propose(trials: &[Trial], config: &SweepConfig, baseline: Option<&Candidate>) -> Proposal {
    propose_with_seen(trials, &HashSet::new(), config, baseline)
}

fn propose_with_seen(
    trials: &[Trial],
    seen: &HashSet<String>,
    config: &SweepConfig,
    baseline: Option<&Candidate>,
) -> Proposal {
    let analysis = analysis::analyze(trials, config);
    optimizer::propose(trials, seen, &mut rng(), config, &analysis, baseline)
}
