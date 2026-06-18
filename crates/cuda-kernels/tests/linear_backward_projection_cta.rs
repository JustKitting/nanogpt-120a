use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use data::assert_vec_close;
use fixture::ProjectionTensors;
use rust_kernels_cuda::linear_backward::LinearBackwardModule;

mod common;
#[path = "linear_backward_projection_cta/data.rs"]
mod data;
#[path = "linear_backward_projection_cta/fixture.rs"]
mod fixture;

const TOKEN_COUNT: usize = 64;
const INPUT_DIM: usize = 64;
const OUTPUT_DIM: usize = 64;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn cta_projection_matches_warp_projection() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        LinearBackwardModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;
    let tensors = ProjectionTensors::new(&stream)?;

    let mut old_dinput = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * INPUT_DIM)?;
    let mut old_dweight = DeviceBuffer::<f32>::zeroed(&stream, OUTPUT_DIM * INPUT_DIM)?;
    let mut cta_dinput = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * INPUT_DIM)?;
    let mut cta_dweight = DeviceBuffer::<f32>::zeroed(&stream, OUTPUT_DIM * INPUT_DIM)?;

    module.backward_device_scale(tensors.args(&stream, &mut old_dinput, &mut old_dweight))?;
    module.backward_device_scale_cta(tensors.args(&stream, &mut cta_dinput, &mut cta_dweight))?;

    assert_vec_close(
        &old_dinput.to_host_vec(&stream)?,
        &cta_dinput.to_host_vec(&stream)?,
    );
    assert_vec_close(
        &old_dweight.to_host_vec(&stream)?,
        &cta_dweight.to_host_vec(&stream)?,
    );
    Ok(())
}
