use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::next_latent::{NextLatGeluBackwardArgs, NextLatModule};

mod common;

const LEN: usize = 257;
const TOLERANCE: f32 = 1.0e-6;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nextlat_gelu_backward_matches_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(NextLatModule::from_module)?;

    let input = values(-4.0, 0.03125);
    let d_out = values(0.25, 0.00390625);
    let input_dev = DeviceBuffer::from_host(&stream, &input)?;
    let d_out_dev = DeviceBuffer::from_host(&stream, &d_out)?;
    let mut d_input_dev = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;

    module.gelu_backward(NextLatGeluBackwardArgs {
        stream: &stream,
        input: &input_dev,
        d_out: &d_out_dev,
        d_input: &mut d_input_dev,
        len: LEN as u32,
    })?;

    let actual = d_input_dev.to_host_vec(&stream)?;
    for index in 0..LEN {
        let expected = d_out[index] * reference_gelu_grad(input[index]);
        assert!(
            (actual[index] - expected).abs() <= TOLERANCE,
            "index={index} actual={} expected={expected}",
            actual[index],
        );
    }
    Ok(())
}

fn values(start: f32, step: f32) -> Vec<f32> {
    (0..LEN).map(|i| start + step * i as f32).collect()
}

fn reference_gelu_grad(x: f32) -> f32 {
    let x2 = x * x;
    let inner = 0.797_884_6 * (x + 0.044_715 * x * x2);
    let tanh = 2.0 / (1.0 + (-2.0 * inner).exp()) - 1.0;
    let inner_grad = 0.797_884_6 * (1.0 + 3.0 * 0.044_715 * x2);
    0.5 * (1.0 + tanh) + 0.5 * x * (1.0 - tanh * tanh) * inner_grad
}
