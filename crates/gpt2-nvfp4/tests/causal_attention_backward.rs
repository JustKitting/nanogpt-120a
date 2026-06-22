use std::error::Error;
use std::path::PathBuf;

use cuda_core::{CudaContext, DeviceBuffer};
use gpt2_nvfp4::{
    AttentionCoreBackwardArgs, GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_QKV, GPT2_SEQ_LEN,
    GPT2_TOKEN_ROWS, HiddenState, QkvActivation,
    causal_attention_backward as gpt2_causal_attention_backward,
};
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionBackwardTcArgs};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;

#[path = "support/attention_core_scratch.rs"]
mod attention_core_scratch;
#[path = "attention_core_backward/data.rs"]
mod data;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn causal_attention_backward_wrapper_matches_direct_kernel() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(ptx_path().as_str())?;
    let module = AttentionModule::from_module(ptx.clone())?;
    let tc_module = F16TcMatmulModule::from_module(ptx)?;

    let qkv = DeviceBuffer::from_host(&stream, &vec![0_u16; QkvActivation::LEN])?;
    let attention_out = DeviceBuffer::from_host(&stream, &vec![0_u16; HiddenState::LEN])?;
    let d_out = DeviceBuffer::from_host(&stream, &data::d_out_values())?;
    let log_sum_exp = DeviceBuffer::from_host(&stream, &data::log_sum_exp_values())?;
    let dummy = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let dummy_u16 = DeviceBuffer::<u16>::zeroed(&stream, 1)?;
    let dummy_bytes = DeviceBuffer::<u8>::zeroed(&stream, 1)?;
    let dummy_scales = DeviceBuffer::<u8>::zeroed(&stream, 1)?;
    let dummy_global_scales = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let saved = data::saved_block(
        &qkv,
        &attention_out,
        &log_sum_exp,
        &dummy,
        &dummy_u16,
        &dummy_bytes,
        &dummy_scales,
        &dummy_global_scales,
    );
    let mut wrapper_d_qkv = DeviceBuffer::<f32>::zeroed(&stream, QkvActivation::LEN)?;
    let mut direct_d_qkv = DeviceBuffer::<f32>::zeroed(&stream, QkvActivation::LEN)?;
    let mut wrapper_scratch = attention_core_scratch::AttentionCoreScratchBuffers::new(&stream)?;
    let mut direct_scratch = attention_core_scratch::AttentionCoreScratchBuffers::new(&stream)?;

    gpt2_causal_attention_backward(AttentionCoreBackwardArgs {
        stream: &stream,
        module: &module,
        tc_module: &tc_module,
        saved,
        d_attention_out: &d_out,
        d_qkv: &mut wrapper_d_qkv,
        scratch: wrapper_scratch.args(),
    })?;

    let direct_core = direct_scratch.args();
    module.causal_attention_backward_tc(CausalAttentionBackwardTcArgs {
        stream: &stream,
        tc_module: &tc_module,
        qkv: &qkv,
        attention_out: &attention_out,
        d_out: &d_out,
        log_sum_exp: &log_sum_exp,
        softmax_d: direct_core.softmax_d,
        d_qkv: &mut direct_d_qkv,
        scratch: direct_core.tc,
        row_count: GPT2_TOKEN_ROWS as u32,
        seq_len: GPT2_SEQ_LEN as u32,
        batch_size: GPT2_BATCH_SIZE as u32,
        embedding_dim: GPT2_N_EMBD as u32,
        qkv_dim: GPT2_QKV as u32,
        head_count: GPT2_N_HEAD as u32,
        head_dim: (GPT2_N_EMBD / GPT2_N_HEAD) as u32,
    })?;

    let wrapper = wrapper_d_qkv.to_host_vec(&stream)?;
    let direct = direct_d_qkv.to_host_vec(&stream)?;
    assert!(wrapper.iter().all(|value| value.is_finite()));
    assert!(wrapper.iter().any(|value| value.abs() > 0.0));
    assert_eq!(
        wrapper
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        direct
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    Ok(())
}

fn gpu_device_index() -> usize {
    std::env::var("CUDA_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn ptx_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../rust_kernels_cuda.ptx")
        .to_string_lossy()
        .into_owned()
}
