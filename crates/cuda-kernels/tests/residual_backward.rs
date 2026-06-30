use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::residual::{ResidualBackwardModule, ResidualGradAddArgs};

mod common;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn residual_grad_add_matches_reference() -> Result<(), Box<dyn Error>> {
    const LEN: usize = 257;
    let direct: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.25 - 8.0).collect();
    let branch: Vec<f32> = (0..LEN).map(|i| 3.0 - i as f32 * 0.125).collect();

    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = ResidualBackwardModule::from_module(ptx)?;

    let direct_dev = DeviceBuffer::from_host(&stream, &direct)?;
    let branch_dev = DeviceBuffer::from_host(&stream, &branch)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;

    module.grad_add(ResidualGradAddArgs {
        stream: &stream,
        direct: &direct_dev,
        branch: &branch_dev,
        out: &mut out_dev,
        len: LEN as u32,
    })?;

    let out = out_dev.to_host_vec(&stream)?;
    for i in 0..LEN {
        assert_eq!(out[i], direct[i] + branch[i]);
    }
    Ok(())
}
