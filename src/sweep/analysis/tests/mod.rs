mod beliefs;
mod scoring;

use super::super::{
    config::SweepConfig,
    history::Trial,
    test_fixtures::{basic_candidate as candidate, quality_config, success_trial as trial},
};
use super::SweepAnalysis;

fn quality_trials() -> [Trial; 4] {
    [
        trial(candidate(4, 4), 9.0),
        trial(candidate(4, 8), 5.0),
        trial(candidate(16, 4), 5.0),
        trial(candidate(16, 8), 1.0),
    ]
}

fn has_model(analysis: &SweepAnalysis, name: &str) -> bool {
    analysis.models.iter().any(|model| model.name == name)
}

fn config() -> SweepConfig {
    quality_config(16)
}
