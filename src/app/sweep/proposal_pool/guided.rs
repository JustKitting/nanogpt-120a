use super::super::{
    candidate::{Candidate, valid_aurora_phases},
    candidate_space,
    rng::SweepRng,
};
use super::direction::Direction;

pub fn candidate(rng: &mut SweepRng, direction: &Direction, jitter: bool) -> Candidate {
    let batch_size = pick(
        &candidate_space::BATCH_SIZE,
        direction.batch_size,
        rng,
        jitter,
    );
    let n_layer = pick(&candidate_space::N_LAYER, direction.n_layer, rng, jitter);
    let (n_embd, n_head) = pick(&candidate_space::N_EMBD, direction.n_embd, rng, jitter);
    let aurora_blocks = pick(
        &candidate_space::AURORA_BLOCKS,
        direction.aurora_blocks,
        rng,
        jitter,
    );
    let phases = valid_aurora_phases(n_layer * 4, aurora_blocks);
    Candidate {
        batch_size,
        n_layer,
        n_embd,
        n_head,
        aurora_phases: pick(&phases, direction.aurora_phases, rng, jitter),
        aurora_blocks,
        lr_scale: pick_f64(
            candidate_space::LR_SCALE_RANGE,
            direction.lr_scale,
            rng,
            jitter,
            true,
        ),
        adam_lr_scale: pick_f64(
            candidate_space::LR_SCALE_RANGE,
            direction.adam_lr_scale,
            rng,
            jitter,
            true,
        ),
        nextlat_lr_scale: pick_f64(
            candidate_space::LR_SCALE_RANGE,
            direction.nextlat_lr_scale,
            rng,
            jitter,
            true,
        ),
        warmup_steps: pick_usize(
            candidate_space::WARMUP_STEPS_RANGE,
            direction.warmup_steps,
            rng,
            jitter,
        ),
        start_ratio: pick_f64(
            candidate_space::START_RATIO_RANGE,
            direction.start_ratio,
            rng,
            jitter,
            false,
        ),
        amuse_beta1: pick_f64(
            candidate_space::AMUSE_BETA1_RANGE,
            direction.amuse_beta1,
            rng,
            jitter,
            false,
        ),
        amuse_rho: pick_f64(
            candidate_space::AMUSE_RHO_RANGE,
            direction.amuse_rho,
            rng,
            jitter,
            false,
        ),
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

fn pick_f64(
    range: (f64, f64),
    direction: f64,
    rng: &mut SweepRng,
    jitter: bool,
    log_scale: bool,
) -> f64 {
    let t = pick_unit(direction, rng, jitter);
    if log_scale {
        let lo = range.0.ln();
        let hi = range.1.ln();
        return (lo + (hi - lo) * t).exp();
    }
    range.0 + (range.1 - range.0) * t
}

fn pick_usize(range: (usize, usize), direction: f64, rng: &mut SweepRng, jitter: bool) -> usize {
    let span = (range.1 - range.0) as f64;
    range.0 + (span * pick_unit(direction, rng, jitter)).round() as usize
}

fn pick_unit(direction: f64, rng: &mut SweepRng, jitter: bool) -> f64 {
    if jitter && rng.f64() < 0.25 {
        return rng.f64();
    }

    let center = if direction > 0.02 {
        0.875
    } else if direction < -0.02 {
        0.125
    } else {
        0.5
    };
    if jitter {
        return (center + (rng.f64() - 0.5) * 0.25).clamp(0.0, 1.0);
    }
    center
}
