#[path = "l3_mlp/assertions.rs"]
mod assertions;
#[path = "l3_mlp/buffers.rs"]
mod buffers;
#[path = "l3_mlp/case.rs"]
mod case;
mod common;
#[path = "l3_mlp/data.rs"]
mod data;
#[path = "common/nvfp4.rs"]
mod nvfp4_common;
#[path = "support/forward_scratch.rs"]
mod scratch_support;
#[path = "l3_mlp/weights.rs"]
mod weights;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn mlp_forward_projects_relu2_downprojects_and_residual_adds(
) -> Result<(), Box<dyn std::error::Error>> {
    case::run()
}
