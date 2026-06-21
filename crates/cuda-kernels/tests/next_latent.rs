use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::next_latent::{NextLatConcatArgs, NextLatModule, NextLatSmoothL1Args};

mod common;

#[path = "next_latent/reference.rs"]
mod reference;

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

    let next_token_embeddings = reference::values(0.25);
    let current_states = reference::values(-0.5);
    let predicted = reference::values(0.125);
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
    let expected_concat = reference::concat(&next_token_embeddings, &current_states);
    let (expected_losses, expected_grad) = reference::smooth_l1(&predicted, &current_states);

    assert_all_close(&concat, &expected_concat);
    assert_all_close(&losses, &expected_losses);
    assert_all_close(&d_pred, &expected_grad);
    Ok(())
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
