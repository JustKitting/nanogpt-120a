//! Device optimizer kernels and launch wrappers.
//!
//! Folder ownership:
//! - `adam`: AdamW updates for scalar/vector weights where Aurora does not apply.
//! - `aurora`: Aurora updates for matrix-shaped weights.
//! - `embedding`: token-embedding gradient scatter from residual gradients.
//! - `schedule_free`: z/x interpolation and materialization for schedule-free state.
//! - `launcher`: host-side CUDA launch wrappers around the device kernels.
//! - `modules`: CUDA module loading registry.
//! - `threads`: shared launch-size constants.

mod adam;
mod args;
mod aurora;
mod embedding;
mod launcher;
mod modules;
mod schedule_free;
mod threads;
mod work_grid;

pub use args::{
    AdamWUpdateArgs, AuroraMegaUpdateArgs, EmbeddingLookupGradArgs, ScheduleFreeAverageArgs,
    ScheduleFreeMaterializeArgs,
};
pub use launcher::OptimizerModule;

pub const AURORA_COOPERATIVE_BLOCKS: usize = 180;
pub const AURORA_MATRIX_PHASES: usize = 8;
