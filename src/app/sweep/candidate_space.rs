use super::{
    candidate::{Candidate, MIN_N_LAYER},
    rng::SweepRng,
};

pub const BATCH_SIZE: [usize; 3] = [4, 8, 16];
pub const N_LAYER: [usize; 2] = [MIN_N_LAYER, 8];
pub const N_EMBD: [(usize, usize); 2] = [(1024, 16), (2048, 16)];
pub const AURORA_BLOCKS: [usize; 5] = [80, 90, 120, 160, 180];
pub const LR_SCALE_RANGE: (f64, f64) = (0.5, 2.5);
pub const WARMUP_STEPS_RANGE: (usize, usize) = (5, 100);
pub const START_RATIO_RANGE: (f64, f64) = (0.0, 0.2);
pub const AMUSE_BETA1_RANGE: (f64, f64) = (0.2, 0.6);
pub const AMUSE_RHO_RANGE: (f64, f64) = (0.5, 1.0);

const AURORA_PHASES: [usize; 4] = [2, 4, 8, 16];

pub fn random(rng: &mut SweepRng) -> Candidate {
    let (n_embd, n_head) = rng.choose(&N_EMBD);
    let n_layer = rng.choose(&N_LAYER);
    let aurora_blocks = rng.choose(&AURORA_BLOCKS);
    let phases = valid_aurora_phases(n_layer * 4, aurora_blocks);
    Candidate {
        batch_size: rng.choose(&BATCH_SIZE),
        n_layer,
        n_embd,
        n_head,
        aurora_phases: rng.choose(&phases),
        aurora_blocks,
        lr_scale: rng.log_uniform(LR_SCALE_RANGE.0, LR_SCALE_RANGE.1),
        adam_lr_scale: rng.log_uniform(LR_SCALE_RANGE.0, LR_SCALE_RANGE.1),
        warmup_steps: random_usize_range(rng, WARMUP_STEPS_RANGE),
        start_ratio: random_f64_range(rng, START_RATIO_RANGE),
        amuse_beta1: random_f64_range(rng, AMUSE_BETA1_RANGE),
        amuse_rho: random_f64_range(rng, AMUSE_RHO_RANGE),
    }
}

pub fn valid_aurora_phases(slots: usize, blocks: usize) -> Vec<usize> {
    AURORA_PHASES
        .into_iter()
        .filter(|phase| slots % phase == 0 && cooperative_blocks(slots, *phase, blocks) <= 360)
        .collect()
}

fn cooperative_blocks(slots: usize, phases: usize, blocks: usize) -> usize {
    blocks * (slots / phases)
}

fn random_f64_range(rng: &mut SweepRng, range: (f64, f64)) -> f64 {
    range.0 + (range.1 - range.0) * rng.f64()
}

fn random_usize_range(rng: &mut SweepRng, range: (usize, usize)) -> usize {
    range.0 + rng.usize(range.1 - range.0 + 1)
}
