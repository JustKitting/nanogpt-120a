use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::optimizer::{
    AdamWUpdateArgs, EmbeddingLookupGradArgs, Nvfp4WeightUpdateArgs, OptimizerModule,
};

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
    let mut chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut global_scale = DeviceBuffer::from_host(&stream, &[1.0_f32])?;

    module.apply_nvfp4_weight_update(Nvfp4WeightUpdateArgs {
        stream: &stream,
        bytes: &mut bytes,
        scales: &mut scales,
        global_scale: &mut global_scale,
        requantize_global_scale: 0.0,
        aurora_update: &update,
        fp32_workspace: &mut workspace,
        amax: &mut amax,
        chunk_amax: &mut chunk_amax,
        len: LEN as u32,
        learning_rate: 0.25,
        weight_decay: 0.1,
    })?;

    let workspace = workspace.to_host_vec(&stream)?;
    let bytes = bytes.to_host_vec(&stream)?;
    let scales = scales.to_host_vec(&stream)?;
    let global_scale = global_scale.to_host_vec(&stream)?;

    assert!(
        workspace
            .iter()
            .all(|value| (*value - 0.85).abs() <= 1.0e-6)
    );
    assert!(bytes.iter().any(|byte| *byte != 0));
    assert!(scales.iter().any(|scale| *scale != 0));
    assert!((global_scale[0] - 0.85 / (256.0 * 6.0)).abs() <= 1.0e-8);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_adamw_update_tracks_moments_and_requantizes() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        OptimizerModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let mut bytes = DeviceBuffer::from_host(&stream, &[E2M1_ONE_PAIR; LEN / 2])?;
    let mut scales = DeviceBuffer::from_host(&stream, &[E4M3_ONE; LEN / 16])?;
    let grad = DeviceBuffer::from_host(&stream, &[0.5_f32; LEN])?;
    let mut first = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let mut second = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let mut residual = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let mut workspace = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let mut amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut global_scale = DeviceBuffer::from_host(&stream, &[1.0_f32])?;

    module.apply_adamw_update(AdamWUpdateArgs {
        stream: &stream,
        bytes: &mut bytes,
        scales: &mut scales,
        global_scale: &mut global_scale,
        requantize_global_scale: 0.0,
        grad: &grad,
        first_moment: &mut first,
        second_moment: &mut second,
        residual: &mut residual,
        fp32_workspace: &mut workspace,
        amax: &mut amax,
        chunk_amax: &mut chunk_amax,
        len: LEN as u32,
        learning_rate: 0.25,
        weight_decay: 0.1,
        beta1: 0.9,
        beta2: 0.95,
        beta1_correction: 0.1,
        beta2_correction: 0.05,
        eps: 1.0e-10,
    })?;

    let workspace = workspace.to_host_vec(&stream)?;
    let first = first.to_host_vec(&stream)?;
    let second = second.to_host_vec(&stream)?;
    assert!(
        workspace
            .iter()
            .all(|value| (*value - 0.725).abs() <= 1.0e-6)
    );
    assert!(first.iter().all(|value| (*value - 0.05).abs() <= 1.0e-6));
    assert!(second.iter().all(|value| (*value - 0.0125).abs() <= 1.0e-6));
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn embedding_lookup_grad_accumulates_duplicate_tokens() -> Result<(), Box<dyn Error>> {
    const TOKEN_COUNT: usize = 3;
    const EMBEDDING_DIM: usize = 4;
    const VOCAB_SIZE: usize = 5;

    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        OptimizerModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let tokens = DeviceBuffer::from_host(&stream, &[2_u32, 2, 3])?;
    let residual = DeviceBuffer::from_host(
        &stream,
        &[
            1.0_f32, 2.0, 3.0, 4.0, //
            0.5, 1.5, 2.5, 3.5, //
            -1.0, -2.0, -3.0, -4.0,
        ],
    )?;
    let mut d_token_embedding = DeviceBuffer::<f32>::zeroed(&stream, VOCAB_SIZE * EMBEDDING_DIM)?;

    module.add_embedding_lookup_grad(EmbeddingLookupGradArgs {
        stream: &stream,
        tokens: &tokens,
        d_embedding_residual: &residual,
        d_token_embedding: &mut d_token_embedding,
        token_count: TOKEN_COUNT as u32,
        embedding_dim: EMBEDDING_DIM as u32,
    })?;

    let actual = d_token_embedding.to_host_vec(&stream)?;
    assert_eq!(&actual[8..12], &[1.5, 3.5, 5.5, 7.5]);
    assert_eq!(&actual[12..16], &[-1.0, -2.0, -3.0, -4.0]);
    Ok(())
}
