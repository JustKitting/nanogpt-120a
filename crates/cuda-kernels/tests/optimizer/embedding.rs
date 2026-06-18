use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::optimizer::{EmbeddingLookupGradArgs, OptimizerModule};

use crate::common;

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
