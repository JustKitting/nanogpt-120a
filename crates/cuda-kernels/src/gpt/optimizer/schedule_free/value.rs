use crate::float_ptx::abs_f32;

#[inline(always)]
pub(super) fn schedule_value(z_master: &[f32], x_master: &[f32], beta: f32, index: u32) -> f32 {
    let i = index as usize;
    let z = z_master[i];
    let x = x_master[i];
    z + beta * (x - z)
}

#[inline(always)]
pub(super) fn checked_abs_schedule_value(
    z_master: &[f32], x_master: &[f32], beta: f32, index: u32, len: u32,
) -> f32 {
    if index < len {
        abs_f32(schedule_value(z_master, x_master, beta, index))
    } else {
        0.0
    }
}
