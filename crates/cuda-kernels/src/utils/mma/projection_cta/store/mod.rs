mod affine;
mod common;
mod nobias;
mod relu2;

pub use affine::store_affine_accumulator;
pub use nobias::store_accumulator;
pub use relu2::store_relu2_accumulator;
