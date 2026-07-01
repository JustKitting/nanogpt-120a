mod burn_shim;
mod config;
mod data_loader;
mod launcher;
mod metrics;
mod output;
mod render;
mod strategy;

pub(super) use burn_shim::CudaLearningComponents;
use burn_shim::{BurnBackend, BurnInnerBackend, CudaTrainInput, CudaValidInput};
pub(super) use config::{env_bool, env_nonempty, TrainConfig};
pub(crate) use launcher::launch_from_env;
pub(super) use metrics::CudaTrainOutput;
