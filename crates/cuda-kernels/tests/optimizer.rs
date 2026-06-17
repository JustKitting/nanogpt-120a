use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::optimizer::{Nvfp4WeightUpdateArgs, OptimizerModule};

mod common;

const LEN: usize = 32;
const E2M1_ONE_PAIR: u8 = 0x22;
const E4M3_ONE: u8 = 0x38;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_weight_update_applies_decay_update_and_requantizes() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        OptimizerModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let mut bytes = DeviceBuffer::from_host(&stream, &[E2M1_ONE_PAIR; LEN / 2])?;
    let mut scales = DeviceBuffer::from_host(&stream, &[E4M3_ONE; LEN / 16])?;
    let update = DeviceBuffer::from_host(&stream, &[0.5_f32; LEN])?;
    let mut workspace = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let mut amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut next_global_scale = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    module.apply_nvfp4_weight_update(Nvfp4WeightUpdateArgs {
        stream: &stream,
        bytes: &mut bytes,
        scales: &mut scales,
        global_scale: 1.0,
        aurora_update: &update,
        fp32_workspace: &mut workspace,
        amax: &mut amax,
        next_global_scale: &mut next_global_scale,
        len: LEN as u32,
        learning_rate: 0.25,
        weight_decay: 0.1,
    })?;

    let workspace = workspace.to_host_vec(&stream)?;
    let bytes = bytes.to_host_vec(&stream)?;
    let scales = scales.to_host_vec(&stream)?;
    let next_global_scale = next_global_scale.to_host_vec(&stream)?;

    assert!(
        workspace
            .iter()
            .all(|value| (*value - 0.85).abs() <= 1.0e-6)
    );
    assert!(bytes.iter().any(|byte| *byte != 0));
    assert!(scales.iter().any(|scale| *scale != 0));
    assert!((next_global_scale[0] - 0.85 / (256.0 * 6.0)).abs() <= 1.0e-8);
    Ok(())
}
