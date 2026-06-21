mod analysis;
mod baseline;
mod candidate;
mod chain;
mod config;
mod history;
mod optimizer;
mod parse;
mod proposal_log;
mod proposal_pool;
mod rng;
mod run_build;
mod run_train;
mod runner;
mod status;
mod trial_row;

#[cfg(test)]
mod tests;

use clap::Parser;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = config::SweepConfig::parse();
    runner::run(config)
}
