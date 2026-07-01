#[path = "l3_mlp/assertions.rs"]
mod assertions;
#[path = "l3_mlp/buffers.rs"]
mod buffers;
#[path = "l3_mlp/case.rs"]
mod case;
mod common;
#[path = "l3_mlp/data.rs"]
mod data;
#[path = "l3_mlp/weights.rs"]
mod weights;

use common::upload::TestResult;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn mlp_forward_projects_relu2_downprojects_and_residual_adds() -> TestResult {
    case::run()
}
