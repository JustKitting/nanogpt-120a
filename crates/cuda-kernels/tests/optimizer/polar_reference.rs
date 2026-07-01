#[path = "polar_reference/metrics.rs"]
mod metrics;
#[path = "polar_reference/normalize.rs"]
mod normalize;
#[path = "polar_reference/ops.rs"]
mod ops;
#[path = "polar_reference/rounding.rs"]
mod rounding;
#[path = "polar_reference/scalar.rs"]
mod scalar;

pub use metrics::{cosine, max_abs_error, relative_l2};
pub use normalize::normalized_polar_source;
pub use ops::{matmul_f16, polar_next};
pub use rounding::round_f16_to_f32;
pub use scalar::polar_first_iteration_scalar;
