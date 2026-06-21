use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::next_latent::{NextLatConcatArgs, NextLatModule, NextLatSmoothL1Args};

mod common;

const BATCH_SIZE: usize = 2;
const SEQ_LEN: usize = 3;
const ROW_COUNT: usize = BATCH_SIZE * SEQ_LEN;
const EMBED: usize = 4;
const LAMBDA: f32 = 2.0;
const TOLERANCE: f32 = 1.0e-7;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nextlat_concat_and_shifted_smooth_l1_match_reference() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        NextLatModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let next_token_embeddings = values(0.25);
    let current_states = values(-0.5);
    let predicted = values(0.125);
    let mut losses = vec![1.0_f32; ROW_COUNT];

    let next_token_dev = DeviceBuffer::from_host(&stream, &next_token_embeddings)?;
    let current_dev = DeviceBuffer::from_host(&stream, &current_states)?;
    let predicted_dev = DeviceBuffer::from_host(&stream, &predicted)?;
    let mut concat_dev = DeviceBuffer::<f32>::zeroed(&stream, ROW_COUNT * EMBED * 2)?;
    let mut losses_dev = DeviceBuffer::from_host(&stream, &losses)?;
    let mut d_pred_dev = DeviceBuffer::<f32>::zeroed(&stream, ROW_COUNT * EMBED)?;

    module.concat_input(NextLatConcatArgs {
        stream: &stream,
        next_token_embeddings: &next_token_dev,
        current_states: &current_dev,
        out: &mut concat_dev,
        row_count: ROW_COUNT as u32,
        embedding_dim: EMBED as u32,
    })?;
    module.smooth_l1(NextLatSmoothL1Args {
        stream: &stream,
        predicted_next_states: &predicted_dev,
        target_states: &current_dev,
        losses: &mut losses_dev,
        d_predicted_next_states: &mut d_pred_dev,
        batch_size: BATCH_SIZE as u32,
        seq_len: SEQ_LEN as u32,
        embedding_dim: EMBED as u32,
        lambda: LAMBDA,
    })?;

    let concat = concat_dev.to_host_vec(&stream)?;
    losses = losses_dev.to_host_vec(&stream)?;
    let d_pred = d_pred_dev.to_host_vec(&stream)?;
    let expected_concat = reference_concat(&next_token_embeddings, &current_states);
    let (expected_losses, expected_grad) = reference_smooth_l1(&predicted, &current_states);

    assert_all_close(&concat, &expected_concat);
    assert_all_close(&losses, &expected_losses);
    assert_all_close(&d_pred, &expected_grad);
    Ok(())
}

fn values(offset: f32) -> Vec<f32> {
    (0..ROW_COUNT * EMBED)
        .map(|index| offset + (index as f32 - 7.0) * 0.0625)
        .collect()
}

fn reference_concat(next_token_embeddings: &[f32], current_states: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; ROW_COUNT * EMBED * 2];
    for row in 0..ROW_COUNT {
        for col in 0..EMBED {
            out[row * EMBED * 2 + col] = next_token_embeddings[row * EMBED + col];
            out[row * EMBED * 2 + EMBED + col] = current_states[row * EMBED + col];
        }
    }
    out
}

fn reference_smooth_l1(predicted: &[f32], target: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut losses = vec![1.0_f32; ROW_COUNT];
    let mut grad = vec![0.0_f32; ROW_COUNT * EMBED];
    let grad_scale = LAMBDA / ((BATCH_SIZE * (SEQ_LEN - 1) * EMBED) as f32);

    for batch in 0..BATCH_SIZE {
        for pos in 0..SEQ_LEN - 1 {
            let row = batch * SEQ_LEN + pos;
            let mut local = 0.0;
            for col in 0..EMBED {
                let offset = row * EMBED + col;
                let target_offset = (row + 1) * EMBED + col;
                let diff = predicted[offset] - target[target_offset];
                let abs = diff.abs();
                let d = if abs < 1.0 {
                    local += 0.5 * diff * diff;
                    diff
                } else {
                    local += abs - 0.5;
                    diff.signum()
                };
                grad[offset] = d * grad_scale;
            }
            losses[row] += LAMBDA * local / EMBED as f32;
        }
    }

    (losses, grad)
}

fn assert_all_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= TOLERANCE,
            "index={index} actual={actual:.8e} expected={expected:.8e} error={error:.8e}"
        );
    }
}
