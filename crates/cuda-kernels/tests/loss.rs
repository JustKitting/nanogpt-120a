use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::loss::{CrossEntropyArgs, LossModule};

mod common;

const TOKEN_COUNT: usize = 2;
const VOCAB_SIZE: usize = 4;
const TOLERANCE: f32 = 1.0e-7;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn cross_entropy_writes_losses_and_dlogits() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = LossModule::from_module(ptx)?;

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
    let mut row_amax_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT)?;

    module.cross_entropy(CrossEntropyArgs {
        stream: &stream,
        logits: &logits_dev,
        targets: &targets_dev,
        losses: &mut losses_dev,
        dlogits: &mut dlogits_dev,
        dlogits_row_amax: &mut row_amax_dev,
        token_count: TOKEN_COUNT as u32,
        vocab_size: VOCAB_SIZE as u32,
    })?;

    let losses = losses_dev.to_host_vec(&stream)?;
    let dlogits = dlogits_dev.to_host_vec(&stream)?;
    let row_amax = row_amax_dev.to_host_vec(&stream)?;
    let expected = expected_loss_and_grad(&logits, &targets);

    common::assert_slice_close(&losses, &expected.0, TOLERANCE);
    common::assert_slice_close(&dlogits, &expected.1, TOLERANCE);

    for (row, actual_amax) in row_amax.iter().enumerate() {
        let base = row * VOCAB_SIZE;
        let expected_amax = expected.1[base..base + VOCAB_SIZE]
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0, f32::max);
        common::assert_close(*actual_amax, expected_amax, TOLERANCE);
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
