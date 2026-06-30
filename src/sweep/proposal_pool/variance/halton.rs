use crate::sweep::candidate_space;

const BASES: [u32; candidate_space::FACTOR_COUNT] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

pub(super) fn units(index: usize) -> [f64; candidate_space::FACTOR_COUNT] {
    std::array::from_fn(|dim| radical_inverse(index, BASES[dim]))
}

fn radical_inverse(mut index: usize, base: u32) -> f64 {
    let base = base as usize;
    let inv_base = 1.0 / base as f64;
    let mut weight = inv_base;
    let mut value = 0.0;
    while index > 0 {
        value += (index % base) as f64 * weight;
        index /= base;
        weight *= inv_base;
    }
    value
}
