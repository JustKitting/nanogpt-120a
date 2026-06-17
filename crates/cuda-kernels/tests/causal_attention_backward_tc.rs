use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionBackwardTcArgs};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;

#[path = "causal_attention_backward_tc/case.rs"]
mod case;
mod common;
#[path = "causal_attention_backward_tc/matmul_scratch.rs"]
mod matmul_scratch;
#[path = "causal_attention_backward_tc/reference.rs"]
mod reference;
#[path = "causal_attention_backward_tc/scratch.rs"]
mod scratch;
#[path = "causal_attention_backward_tc/shape.rs"]
mod shape;

use scratch::TcScratchBuffers;

const TC_TOLERANCE: f32 = 1.0e-6;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn materialized_tc_backward_matches_reference() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
    let attention = AttentionModule::from_module(ptx.clone())?;
    let tc = F16TcMatmulModule::from_module(ptx)?;
    let case = case::simple_case();

    let qkv = DeviceBuffer::from_host(&stream, &case.qkv)?;
    let out = DeviceBuffer::from_host(&stream, &case.out)?;
    let d_out = DeviceBuffer::from_host(&stream, &case.d_out)?;
    let log_sum_exp = DeviceBuffer::from_host(&stream, &case.log_sum_exp)?;
    let mut tc_softmax_d = DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::HEADS)?;
    let mut tc_grad = DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::QKV_DIM)?;
    let mut scratch = TcScratchBuffers::new(&stream)?;
    attention.causal_attention_backward_tc(CausalAttentionBackwardTcArgs {
        stream: &stream,
        tc_module: &tc,
        qkv: &qkv,
        attention_out: &out,
        d_out: &d_out,
        log_sum_exp: &log_sum_exp,
        softmax_d: &mut tc_softmax_d,
        d_qkv: &mut tc_grad,
        scratch: scratch.args(),
        row_count: shape::TOKEN_COUNT as u32,
        seq_len: shape::TOKEN_COUNT as u32,
        batch_size: 1,
        embedding_dim: shape::EMBEDDING as u32,
        qkv_dim: shape::QKV_DIM as u32,
        head_count: shape::HEADS as u32,
        head_dim: shape::HEAD_DIM as u32,
    })?;

    assert_tc_close(&tc_grad.to_host_vec(&stream)?, &case.expected);
    Ok(())
}

fn assert_tc_close(actual: &[f32], expected: &[f32]) {
    let mut max_error = 0.0_f32;
    let mut max_index = 0_usize;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        if error > max_error {
            max_error = error;
            max_index = index;
        }
    }
    assert!(
        max_error <= TC_TOLERANCE,
        "max_error={max_error:.8e} index={max_index} actual={:.8e} expected={:.8e}",
        actual[max_index],
        expected[max_index],
    );
}
