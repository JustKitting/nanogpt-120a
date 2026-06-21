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
pub const FACTOR_COUNT: usize = 11;

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

pub fn from_unit(unit: [f64; FACTOR_COUNT]) -> Candidate {
    let (n_embd, n_head) = choose_unit(&N_EMBD, unit[2]);
    let n_layer = choose_unit(&N_LAYER, unit[1]);
    let aurora_blocks = choose_unit(&AURORA_BLOCKS, unit[4]);
    let phases = valid_aurora_phases(n_layer * 4, aurora_blocks);
    Candidate {
        batch_size: choose_unit(&BATCH_SIZE, unit[0]),
        n_layer,
        n_embd,
        n_head,
        aurora_phases: choose_unit(&phases, unit[3]),
        aurora_blocks,
        lr_scale: log_lerp(LR_SCALE_RANGE, unit[5]),
        adam_lr_scale: log_lerp(LR_SCALE_RANGE, unit[6]),
        warmup_steps: range_usize(WARMUP_STEPS_RANGE, unit[7]),
        start_ratio: range_f64(START_RATIO_RANGE, unit[8]),
        amuse_beta1: range_f64(AMUSE_BETA1_RANGE, unit[9]),
        amuse_rho: range_f64(AMUSE_RHO_RANGE, unit[10]),
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

fn choose_unit<T: Copy>(values: &[T], unit: f64) -> T {
    let index = (unit.clamp(0.0, 1.0) * values.len() as f64).floor() as usize;
    values[index.min(values.len() - 1)]
}

fn log_lerp(range: (f64, f64), unit: f64) -> f64 {
    let lo = range.0.ln();
    let hi = range.1.ln();
    (lo + (hi - lo) * unit.clamp(0.0, 1.0)).exp()
}

fn range_f64(range: (f64, f64), unit: f64) -> f64 {
    range.0 + (range.1 - range.0) * unit.clamp(0.0, 1.0)
}

fn range_usize(range: (usize, usize), unit: f64) -> usize {
    let span = (range.1 - range.0) as f64;
    range.0 + (span * unit.clamp(0.0, 1.0)).round() as usize
}

#[cfg(test)]
mod tests {
    use super::{FACTOR_COUNT, from_unit, valid_aurora_phases};

    #[test]
    fn unit_mapping_keeps_candidate_in_valid_space() {
        let low = from_unit([0.0; FACTOR_COUNT]);
        let high = from_unit([1.0; FACTOR_COUNT]);

        assert_eq!(low.batch_size, 4);
        assert_eq!(low.n_layer, 4);
        assert_eq!(low.n_embd, 1024);
        assert_eq!(high.batch_size, 16);
        assert_eq!(high.n_layer, 8);
        assert_eq!(high.n_embd, 2048);
        assert!(
            valid_aurora_phases(low.n_layer * 4, low.aurora_blocks).contains(&low.aurora_phases)
        );
        assert!(
            valid_aurora_phases(high.n_layer * 4, high.aurora_blocks).contains(&high.aurora_phases)
        );
    }
}
