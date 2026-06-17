mod output_projection;
mod transforms;
mod types;

pub use output_projection::c_proj_backward;
pub use types::{
    AttentionBackwardModules, AttentionBackwardSeeds, AttentionCProjBackwardArgs,
    AttentionCProjScratch,
};
