use crate::sweep::rng::SweepRng;

pub(in crate::sweep) fn rng() -> SweepRng {
    SweepRng::new(0x4750_5432)
}
