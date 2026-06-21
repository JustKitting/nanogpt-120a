use super::super::{
    candidate::{Candidate, MIN_N_LAYER, valid_aurora_phases},
    rng::SweepRng,
};
use super::direction::Direction;

const BATCH_SIZE: [usize; 3] = [4, 8, 16];
const N_LAYER: [usize; 2] = [MIN_N_LAYER, 8];
const N_EMBD: [(usize, usize); 2] = [(1024, 16), (2048, 16)];
const AURORA_BLOCKS: [usize; 5] = [80, 90, 120, 160, 180];
const LR_SCALE: [f64; 3] = [0.5, 1.0, 2.5];
const WARMUP_STEPS: [usize; 4] = [5, 20, 50, 100];
const START_RATIO: [f64; 4] = [0.0, 0.05, 0.1, 0.2];
const AMUSE_BETA1: [f64; 3] = [0.2, 0.4, 0.6];
const AMUSE_RHO: [f64; 3] = [0.5, 0.8, 1.0];

pub fn candidate(rng: &mut SweepRng, direction: &Direction, jitter: bool) -> Candidate {
    let batch_size = pick(&BATCH_SIZE, direction.batch_size, rng, jitter);
    let n_layer = pick(&N_LAYER, direction.n_layer, rng, jitter);
    let (n_embd, n_head) = pick(&N_EMBD, direction.n_embd, rng, jitter);
    let aurora_blocks = pick(&AURORA_BLOCKS, direction.aurora_blocks, rng, jitter);
    let phases = valid_aurora_phases(n_layer * 4, aurora_blocks);
    Candidate {
        batch_size,
        n_layer,
        n_embd,
        n_head,
        aurora_phases: pick(&phases, direction.aurora_phases, rng, jitter),
        aurora_blocks,
        lr_scale: pick(&LR_SCALE, direction.lr_scale, rng, jitter),
        adam_lr_scale: pick(&LR_SCALE, direction.adam_lr_scale, rng, jitter),
        warmup_steps: pick(&WARMUP_STEPS, direction.warmup_steps, rng, jitter),
        start_ratio: pick(&START_RATIO, direction.start_ratio, rng, jitter),
        amuse_beta1: pick(&AMUSE_BETA1, direction.amuse_beta1, rng, jitter),
        amuse_rho: pick(&AMUSE_RHO, direction.amuse_rho, rng, jitter),
    }
}

fn pick<T: Copy>(values: &[T], direction: f64, rng: &mut SweepRng, jitter: bool) -> T {
    if jitter && rng.f64() < 0.25 {
        return rng.choose(values);
    }
    let index = if direction > 0.02 {
        values.len() - 1
    } else if direction < -0.02 {
        0
    } else {
        values.len() / 2
    };
    values[index]
}
