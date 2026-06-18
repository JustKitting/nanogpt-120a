use cuda_device::{DisjointSlice, cuda_module, kernel};

use super::linear_combination;

#[allow(static_mut_refs)]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    pub fn elementwise_linear_combination_kernel(
        a: &[f32],
        b: &[f32],
        mut out: DisjointSlice<f32>,
        a_scale: f32,
        b_scale: f32,
        len: u32,
    ) {
        linear_combination::elementwise_linear_combination(a, b, &mut out, a_scale, b_scale, len);
    }
}
