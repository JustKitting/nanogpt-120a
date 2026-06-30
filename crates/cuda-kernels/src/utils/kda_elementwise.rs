#[path = "kda_elementwise/context.rs"]
mod context;
#[path = "kda_elementwise/prepare.rs"]
mod prepare;

pub(crate) use {context::*, prepare::*};
