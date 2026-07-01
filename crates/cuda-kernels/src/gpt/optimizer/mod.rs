//! Device optimizer kernels and launch wrappers.
//!
//! Folder ownership:
//! - `adam`: AdamW updates for scalar/vector weights where Aurora does not apply.
//! - `aurora`: Aurora updates for matrix-shaped weights.
//! - `embedding`: token-embedding gradient scatter from residual gradients.
//! - `grad_clip`: global-norm clipping over parameter-gradient buffers.
//! - `schedule_free`: z/x interpolation and materialization for schedule-free state.
//! - `launcher`: host-side CUDA launch wrappers around the device kernels.
//! - `modules`: CUDA module loading registry.
//! - `threads`: shared launch-size constants.

mod adam;
mod args;
mod aurora;
mod embedding;
mod grad_clip;
mod kda_clip;
mod launcher;
mod modules;
mod schedule_free;
mod threads;
mod work_grid;

pub use args::{
    AdamWUpdateArgs, AuroraMegaUpdateArgs, AuroraSlotDescriptor, AuroraTmaFinishArgs,
    AuroraTmaPrepareArgs, EmbeddingLookupGradArgs, GradientClipArgs, KdaAuroraClipArgs,
    ScheduleFreeMaterializeArgs,
};
pub use aurora::polar::fused::{
    Coefficients as AuroraPolarCoefficients, coefficients as aurora_polar_coefficients,
};
pub use grad_clip::GRAD_CLIP_VALUES_PER_CHUNK;
pub use launcher::OptimizerModule;

include!(concat!(env!("OUT_DIR"), "/optimizer_config.rs"));
