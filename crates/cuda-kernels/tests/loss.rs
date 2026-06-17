use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::loss::{CrossEntropyArgs, LossModule};

mod common;

const TOKEN_COUNT: usize = 2;
const VOCAB_SIZE: usize = 4;
const TOLERANCE: f32 = 1.0e-7;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn cross_entropy_writes_losses_and_dlogits() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module = LossModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let logits = [
        0.0_f32,
        f32::NEG_INFINITY,
        0.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        0.0,
        f32::NEG_INFINITY,
        0.0,
    ];
    let targets = [0_u32, 3];
    let logits_dev = DeviceBuffer::from_host(&stream, &logits)?;
    let targets_dev = DeviceBuffer::from_host(&stream, &targets)?;
    let mut losses_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT)?;
    let mut dlogits_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * VOCAB_SIZE)?;

    module.cross_entropy(CrossEntropyArgs {
        stream: &stream,
        logits: &logits_dev,
        targets: &targets_dev,
        losses: &mut losses_dev,
        dlogits: &mut dlogits_dev,
        token_count: TOKEN_COUNT as u32,
        vocab_size: VOCAB_SIZE as u32,
    })?;

    let losses = losses_dev.to_host_vec(&stream)?;
    let dlogits = dlogits_dev.to_host_vec(&stream)?;
    let expected = expected_loss_and_grad(&logits, &targets);

    for (actual, expected) in losses.iter().zip(expected.0.iter()) {
        assert_close(*actual, *expected);
    }

    for (actual, expected) in dlogits.iter().zip(expected.1.iter()) {
        assert_close(*actual, *expected);
    }

    Ok(())
}

fn expected_loss_and_grad(logits: &[f32], targets: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let mut losses = vec![0.0_f32; TOKEN_COUNT];
    let mut grad = vec![0.0_f32; TOKEN_COUNT * VOCAB_SIZE];

    for row in 0..TOKEN_COUNT {
        let row_base = row * VOCAB_SIZE;
        let row_logits = &logits[row_base..row_base + VOCAB_SIZE];
        let row_max = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = row_logits
            .iter()
            .map(|value| (*value - row_max).exp())
            .sum::<f32>();
        let target = targets[row] as usize;
        losses[row] = denom.ln() + row_max - row_logits[target];

        for col in 0..VOCAB_SIZE {
            let probability = (row_logits[col] - row_max).exp() / denom;
            grad[row_base + col] =
                (probability - if col == target { 1.0 } else { 0.0 }) / TOKEN_COUNT as f32;
        }
    }

    (losses, grad)
}

fn assert_close(actual: f32, expected: f32) {
    let error = (actual - expected).abs();
    assert!(
        error <= TOLERANCE,
        "actual={actual:.8e} expected={expected:.8e} error={error:.8e} tolerance={TOLERANCE:.8e}"
    );
}
