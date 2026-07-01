mod algorithms;
mod cost;
mod data;
mod update;

pub use crate::polar_reference::{cosine, normalized_polar_source, relative_l2};
pub use algorithms::{stabilized_gram_ns, standard_polar};
pub use cost::{stabilized_gram_ns_cost, standard_cost};
pub use data::gradient;
pub use update::first_iteration_update;
