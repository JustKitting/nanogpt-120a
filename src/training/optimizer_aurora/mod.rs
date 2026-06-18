//! Host-side Aurora update flow.

mod apply;
mod groups;

pub(super) use apply::{AuroraMegaArgs, apply_aurora_mega};
pub(super) use groups::{AuroraGroupTable, AuroraPointerTables};

const MU: f32 = 0.95;
const POLAR_ITERATIONS: u32 = 5;
pub(super) const AURORA_LR: f32 = 1.0e-4;
pub(super) const AURORA_WEIGHT_DECAY: f32 = 0.025;

pub(super) fn aurora_learning_rate(step: u32) -> f32 {
    AURORA_LR * super::learning_rate::aurora_multiplier(step)
}
