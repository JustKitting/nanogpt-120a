use cuda_device::DisjointSlice;

use crate::float_ptx::abs_f32;

#[inline(always)]
pub(super) fn update_average_abs(
    z_master: &mut DisjointSlice<f32>,
    x_master: &mut DisjointSlice<f32>,
    aurora_update: &[f32],
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    index: u32,
    len: u32,
) -> f32 {
    if index >= len {
        return 0.0;
    }

    let i = index as usize;
    unsafe {
        let z_master = z_master.as_mut_ptr().add(i);
        let x_master = x_master.as_mut_ptr().add(i);
        let z = *z_master;
        let x = *x_master;
        let decay = 1.0 - learning_rate * weight_decay;
        let next_z = z * decay - learning_rate * aurora_update[i];
        let next_x = x + average_coefficient * (next_z - x);
        *z_master = next_z;
        *x_master = next_x;
        abs_f32(next_x)
    }
}
