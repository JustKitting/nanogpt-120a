use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::next_latent::{NextLatConcatBackwardArgs, NextLatModule};

mod common;

const ROW_COUNT: usize = 6;
const EMBED: usize = 4;
const TOLERANCE: f32 = 1.0e-7;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nextlat_concat_backward_splits_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(NextLatModule::from_module)?;

    let d_concat = values(ROW_COUNT * EMBED * 2, 0.375);
    let d_predicted = values(ROW_COUNT * EMBED, -0.25);
    let d_concat_dev = DeviceBuffer::from_host(&stream, &d_concat)?;
    let d_predicted_dev = DeviceBuffer::from_host(&stream, &d_predicted)?;
    let mut d_next_dev = DeviceBuffer::<f32>::zeroed(&stream, ROW_COUNT * EMBED)?;
    let mut d_current_dev = DeviceBuffer::<f32>::zeroed(&stream, ROW_COUNT * EMBED)?;

    module.concat_backward(NextLatConcatBackwardArgs {
        stream: &stream,
        d_concat: &d_concat_dev,
        d_predicted: &d_predicted_dev,
        d_next_token_embeddings: &mut d_next_dev,
        d_current_states: &mut d_current_dev,
        row_count: ROW_COUNT as u32,
        embedding_dim: EMBED as u32,
    })?;

    let d_next = d_next_dev.to_host_vec(&stream)?;
    let d_current = d_current_dev.to_host_vec(&stream)?;
    let (expected_next, expected_current) = reference(&d_concat, &d_predicted);

    common::assert_slice_close(&d_next, &expected_next, TOLERANCE);
    common::assert_slice_close(&d_current, &expected_current, TOLERANCE);
    Ok(())
}

fn values(len: usize, start: f32) -> Vec<f32> {
    (0..len).map(|i| start + i as f32 * 0.03125).collect()
}

fn reference(d_concat: &[f32], d_predicted: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut d_next = vec![0.0; ROW_COUNT * EMBED];
    let mut d_current = vec![0.0; ROW_COUNT * EMBED];
    for row in 0..ROW_COUNT {
        for col in 0..EMBED {
            let out = row * EMBED + col;
            let concat = row * EMBED * 2 + col;
            d_next[out] = d_concat[concat];
            d_current[out] = d_concat[concat + EMBED] + d_predicted[out];
        }
    }
    (d_next, d_current)
}
