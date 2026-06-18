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

pub use args::{
    AdamWUpdateArgs, EmbeddingLookupGradArgs, Nvfp4WeightUpdateArgs, ScheduleFreeAverageArgs,
    ScheduleFreeMaterializeArgs,
};
pub use launcher::OptimizerModule;

pub const POLAR_SUM_VALUES_PER_BLOCK: usize = 1024;

pub fn polar_normalize_chunks(element_count: usize) -> usize {
    element_count.div_ceil(POLAR_SUM_VALUES_PER_BLOCK)
}
