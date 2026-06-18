use std::time::Instant;

pub(super) fn seed(step: u32, salt: u32) -> u32 {
    step.wrapping_mul(0x9e37_79b9) ^ salt
}

pub(super) fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}
