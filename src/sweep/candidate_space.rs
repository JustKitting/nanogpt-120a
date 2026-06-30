mod sample;
mod unit;

use super::candidate::MIN_N_LAYER;

pub(in crate::sweep) use sample::{random, valid_aurora_phases};
pub(in crate::sweep) use unit::{choose_unit, from_unit, log_lerp, range_f64, range_usize};

pub const BATCH_SIZE: [usize; 8] = [4, 8, 12, 16, 20, 24, 28, 32];
pub const N_LAYER: [usize; 2] = [MIN_N_LAYER, 8];
pub const N_EMBD: [(usize, usize); 2] = [(1024, 16), (2048, 16)];
pub const AURORA_BLOCKS: [usize; 5] = [80, 90, 120, 160, 180];
pub const LR_SCALE_RANGE: (f64, f64) = (0.5, 2.5);
pub const WARMUP_STEPS_RANGE: (usize, usize) = (5, 100);
pub const START_RATIO_RANGE: (f64, f64) = (0.0, 0.2);
pub const AMUSE_BETA1_RANGE: (f64, f64) = (0.2, 0.6);
pub const AMUSE_RHO_RANGE: (f64, f64) = (0.5, 1.0);
pub const FACTOR_COUNT: usize = 12;
