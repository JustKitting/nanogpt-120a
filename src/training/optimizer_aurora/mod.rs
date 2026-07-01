//! Host-side Aurora update flow.

mod groups;
mod tma;

use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD, GPT2_N_LAYER, GPT2_QKV, NEXTLAT_HIDDEN, NEXTLAT_INPUT};

pub(super) use groups::{AuroraGroupTable, AuroraPointerTables};
pub(super) use tma::{AuroraTmaArgs, apply_aurora_tma};

const MU: f32 = 0.95;
const POLAR_ITERATIONS: u32 = 5;
pub(super) const AURORA_LR: f32 = 1.0e-4;
pub(super) const AURORA_WEIGHT_DECAY: f32 = 0.025;
pub(in crate::training) const AURORA_MATRIX_SLOTS: usize = GPT2_N_LAYER * 4 + 3;

pub(super) fn aurora_learning_rate(step: u32) -> f32 {
    AURORA_LR * super::learning_rate::aurora_multiplier(step)
}

pub(in crate::training) const fn max_matrix_len() -> usize {
    max3(
        GPT2_N_EMBD * GPT2_QKV,
        GPT2_MLP * GPT2_N_EMBD,
        NEXTLAT_INPUT * NEXTLAT_HIDDEN,
    )
}

pub(in crate::training) const fn max_matrix_dim() -> usize {
    max2(GPT2_N_EMBD, NEXTLAT_HIDDEN)
}

pub(in crate::training) const fn max_polar_cols() -> usize {
    max3(max2(GPT2_QKV, GPT2_MLP), NEXTLAT_INPUT, NEXTLAT_HIDDEN)
}

const fn max2(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

const fn max3(a: usize, b: usize, c: usize) -> usize {
    max2(max2(a, b), c)
}
