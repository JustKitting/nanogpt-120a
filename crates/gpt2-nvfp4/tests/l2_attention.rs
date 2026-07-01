use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionForwardArgs, AttentionLogSumExp, AttentionProjectionTensors, AttentionWeights,
    HiddenState, HiddenStateDevice, HiddenVectorShape, QkvActivation, QkvVectorShape,
    QkvWeightShape, ResidualWeightShape, RowwiseNvfp4Buffers, GPT2_CONTEXT_LEN, GPT2_N_HEAD,
};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

#[path = "l2_attention/assertions.rs"]
mod assertions;
mod common;
#[path = "l2_attention/data.rs"]
mod data;

use assertions::{
    assert_attention_log_sum_exp, assert_attention_matches, assert_c_proj_residual_add,
    assert_output_amax, assert_qkv_nonzero,
};
use common::cuda_test_context;
use common::forward_scratch::CausalAttentionTcScratchBuffers;
use common::upload::{upload_nvfp4_bytes, upload_zero_nvfp4, TestResult};
use data::{c_proj_identity_weight_bytes, hidden_input, qkv_identity_weight_bytes, residual_input};

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn attention_forward_quantizes_projects_and_applies_causal_attention() -> TestResult {
    let (_, stream, module) = cuda_test_context()?;
    let attention_module = AttentionModule::from_module(module.clone())?;
    let tc_module = F16TcMatmulModule::from_module(module.clone())?;
    let quant_module = Nvfp4QuantModule::from_module(module)?;

    let (hidden, amax) = hidden_input();
    let residual = residual_input();
    let mut residual_dev = DeviceBuffer::from_host(&stream, &residual)?;
    let mut hidden_dev = DeviceBuffer::from_host(&stream, &hidden)?;
    let mut amax_dev = DeviceBuffer::from_host(&stream, &amax)?;
    let mut mean_dev = DeviceBuffer::<f32>::zeroed(&stream, GPT2_CONTEXT_LEN)?;
    let mut inv_std_dev = DeviceBuffer::<f32>::zeroed(&stream, GPT2_CONTEXT_LEN)?;
    let mut input_nvfp4 = RowwiseNvfp4Buffers::new(&stream, HiddenState::LEN, GPT2_CONTEXT_LEN)?;
    let mut qkv_dev = DeviceBuffer::<f32>::zeroed(&stream, QkvActivation::LEN)?;
    let mut attention_log_sum_exp_dev =
        DeviceBuffer::<f32>::zeroed(&stream, AttentionLogSumExp::LEN)?;
    let mut tc_scratch = CausalAttentionTcScratchBuffers::new(
        &stream,
        HiddenState::LEN,
        1,
        GPT2_N_HEAD,
        GPT2_CONTEXT_LEN,
    )?;

    let qkv_weight = upload_nvfp4_bytes::<QkvWeightShape>(&stream, qkv_identity_weight_bytes())?;
    let qkv_bias = upload_zero_nvfp4::<QkvVectorShape>(&stream)?;
    let c_proj_weight =
        upload_nvfp4_bytes::<ResidualWeightShape>(&stream, c_proj_identity_weight_bytes())?;
    let c_proj_bias = upload_zero_nvfp4::<HiddenVectorShape>(&stream)?;

    AttentionWeights::forward(AttentionForwardArgs {
        use_full_attention: true,
        module: &attention_module,
        tc_module: &tc_module,
        quant_module: &quant_module,
        input_nvfp4: input_nvfp4.scratch(),
        tc_scratch: tc_scratch.args(),
        projections: AttentionProjectionTensors {
            qkv_weight: qkv_weight.mma(),
            qkv_bias: qkv_bias.device(),
            c_proj_weight: c_proj_weight.mma(),
            c_proj_bias: c_proj_bias.device(),
        },
        qkv: &mut qkv_dev,
        attention_log_sum_exp: &mut attention_log_sum_exp_dev,
        hidden: HiddenStateDevice {
            stream: &stream,
            batch_size: 1,
            seq_len: GPT2_CONTEXT_LEN as u32,
            row_count: GPT2_CONTEXT_LEN as u32,
            residual: &mut residual_dev,
            normalized: &mut hidden_dev,
            normalized_amax: &mut amax_dev,
            mean: &mut mean_dev,
            inv_std: &mut inv_std_dev,
        },
        tape: None,
    })?;

    let qkv = qkv_dev.to_host_vec(&stream)?;
    let out = hidden_dev.to_host_vec(&stream)?;
    let attention_log_sum_exp = attention_log_sum_exp_dev.to_host_vec(&stream)?;
    let output_amax = amax_dev.to_host_vec(&stream)?;
    let residual_out = residual_dev.to_host_vec(&stream)?;
    assert_qkv_nonzero(&qkv);
    assert_attention_log_sum_exp(&attention_log_sum_exp);
    assert_attention_matches(&qkv, &out);
    assert_output_amax(&out, &output_amax);
    assert_c_proj_residual_add(&residual, &out, &residual_out);
    Ok(())
}
