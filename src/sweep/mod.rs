mod analysis;
mod baseline;
mod candidate;
mod candidate_space;
mod chain;
mod config;
mod features;
mod fmt;
mod history;
mod optimizer;
mod parse;
mod proposal_log;
mod proposal_pool;
mod rng;
mod run_build;
mod run_train;
mod runner;
mod screen_gate;
mod status;
#[cfg(test)]
mod test_fixtures;
mod trial_row;

#[cfg(test)]
mod tests;

use clap::Parser;

pub(crate) type SweepResult<T = ()> = Result<T, Box<dyn std::error::Error>>;
pub fn run() -> SweepResult {
    let config = config::SweepConfig::parse();
    runner::run(config)
}
