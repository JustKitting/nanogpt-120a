use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionBackwardArgs};

#[path = "causal_attention_backward/check.rs"]
mod check;
mod common;
#[path = "causal_attention_backward/reference.rs"]
mod reference;
#[path = "causal_attention_backward/reference_math.rs"]
mod reference_math;
#[path = "causal_attention_backward/shape.rs"]
mod shape;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn causal_attention_backward_matches_rope_reference() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        AttentionModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;
    let case = reference::case();

    let qkv = DeviceBuffer::from_host(&stream, &case.qkv)?;
    let out = DeviceBuffer::from_host(&stream, &case.out)?;
    let d_out = DeviceBuffer::from_host(&stream, &case.d_out)?;
    let log_sum_exp = DeviceBuffer::from_host(&stream, &case.log_sum_exp)?;
    let mut softmax_d = DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::HEADS)?;
    let mut d_qkv = DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::QKV_DIM)?;

    module.causal_attention_backward(CausalAttentionBackwardArgs {
        stream: &stream,
        qkv: &qkv,
        attention_out: &out,
        d_out: &d_out,
        log_sum_exp: &log_sum_exp,
        softmax_d: &mut softmax_d,
        d_qkv: &mut d_qkv,
        row_count: shape::TOKEN_COUNT as u32,
        seq_len: shape::TOKEN_COUNT as u32,
        batch_size: 1,
        embedding_dim: shape::EMBEDDING as u32,
        qkv_dim: shape::QKV_DIM as u32,
        head_count: shape::HEADS as u32,
        head_dim: shape::HEAD_DIM as u32,
    })?;

    let actual = d_qkv.to_host_vec(&stream)?;
    check::assert_grad_close(&actual, &case.expected);
    Ok(())
}
