use crate::float_ptx::{abs_f32, max_f32};

#[inline(always)]
pub fn max4_f32(a: f32, b: f32, c: f32, d: f32) -> f32 {
    max_f32(max_f32(a, b), max_f32(c, d))
}

#[inline(always)]
pub fn amax4_f32(a: f32, b: f32, c: f32, d: f32) -> f32 {
    max4_f32(abs_f32(a), abs_f32(b), abs_f32(c), abs_f32(d))
}
