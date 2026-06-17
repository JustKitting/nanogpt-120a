use std::error::Error;
use std::path::PathBuf;

use cuda_core::{CudaContext, DeviceBuffer};
use gpt2_nvfp4::{
    AttentionCoreBackwardArgs, AttentionCoreScratch, AttentionLse, GPT2_CONTEXT_LEN, GPT2_N_EMBD,
    GPT2_N_HEAD, GPT2_QKV, HiddenState, QkvActivation,
    causal_attention_backward as gpt2_causal_attention_backward,
};
use rust_kernels_cuda::attention::{
    AttentionModule, CausalAttentionBackwardArgs as CudaCausalAttentionBackwardArgs,
};

#[path = "causal_attention_backward/data.rs"]
mod data;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn causal_attention_backward_wrapper_matches_direct_kernel() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module = AttentionModule::from_module(ctx.load_module_from_file(ptx_path().as_str())?)?;

    let qkv = DeviceBuffer::from_host(&stream, &vec![0.0_f32; QkvActivation::LEN])?;
    let attention_out = DeviceBuffer::from_host(&stream, &vec![0.0_f32; HiddenState::LEN])?;
    let d_out = DeviceBuffer::from_host(&stream, &data::d_out_values())?;
    let lse = DeviceBuffer::from_host(&stream, &data::lse_values())?;
    let dummy = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let dummy_bytes = DeviceBuffer::<u8>::zeroed(&stream, 1)?;
    let dummy_scales = DeviceBuffer::<u8>::zeroed(&stream, 1)?;
    let dummy_global_scales = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let saved = data::saved_block(
        &qkv,
        &attention_out,
        &lse,
        &dummy,
        &dummy_bytes,
        &dummy_scales,
        &dummy_global_scales,
    );
    let mut wrapper_softmax_d = DeviceBuffer::<f32>::zeroed(&stream, AttentionLse::LEN)?;
    let mut direct_softmax_d = DeviceBuffer::<f32>::zeroed(&stream, AttentionLse::LEN)?;
    let mut wrapper_d_qkv = DeviceBuffer::<f32>::zeroed(&stream, QkvActivation::LEN)?;
    let mut direct_d_qkv = DeviceBuffer::<f32>::zeroed(&stream, QkvActivation::LEN)?;

    gpt2_causal_attention_backward(AttentionCoreBackwardArgs {
        stream: &stream,
        module: &module,
        saved,
        d_attention_out: &d_out,
        d_qkv: &mut wrapper_d_qkv,
        scratch: AttentionCoreScratch {
            softmax_d: &mut wrapper_softmax_d,
        },
    })?;

    module.causal_attention_backward(CudaCausalAttentionBackwardArgs {
        stream: &stream,
        qkv: &qkv,
        attention_out: &attention_out,
        d_out: &d_out,
        lse: &lse,
        softmax_d: &mut direct_softmax_d,
        d_qkv: &mut direct_d_qkv,
        token_count: GPT2_CONTEXT_LEN as u32,
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
