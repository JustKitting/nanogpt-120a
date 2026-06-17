pub const QUARTET_MS_EDEN_SCALE_OVERRIDE: f32 = (17.0 / 16.0) * 0.93;
pub const QUARTET_MS_EDEN_FP8_MAX: f32 = 256.0;
pub const QUARTET_MS_EDEN_FP4_MAX: f32 = 6.0;

#[inline(always)]
pub fn quartet_backward_ms_eden_global_scale(amax: f32) -> f32 {
    if amax == 0.0 {
        1.0
    } else {
        amax * QUARTET_MS_EDEN_SCALE_OVERRIDE / (QUARTET_MS_EDEN_FP8_MAX * QUARTET_MS_EDEN_FP4_MAX)
    }
}
