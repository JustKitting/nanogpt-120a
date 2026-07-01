use std::error::Error;

use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionForwardArgs, AttentionLogSumExp, AttentionProjectionTensors, AttentionWeights,
    GPT2_CONTEXT_LEN, GPT2_N_HEAD, HiddenState, HiddenStateDevice, HiddenStateNvfp4,
    HiddenVectorShape, Nvfp4Shape, QkvActivation, QkvVectorShape, QkvWeightShape,
    ResidualWeightShape,
};
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionTcScratch};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

#[path = "l2_attention/assertions.rs"]
mod assertions;
mod common;
#[path = "l2_attention/data.rs"]
mod data;
#[path = "common/f16.rs"]
mod f16_common;
#[path = "common/nvfp4.rs"]
mod nvfp4_common;

use assertions::{
    assert_attention_log_sum_exp, assert_attention_matches, assert_c_proj_residual_add,
    assert_output_amax, assert_qkv_nonzero,
};
use common::cuda_test_context;
use data::{c_proj_identity_weight_bytes, hidden_input, qkv_identity_weight_bytes, residual_input};
use nvfp4_common::filled_u8;

const E4M3_ONE: u8 = 0x38;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn attention_forward_quantizes_projects_and_applies_causal_attention() -> Result<(), Box<dyn Error>>
{
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
    let mut input_bytes_dev = DeviceBuffer::<u8>::zeroed(&stream, HiddenState::LEN / 2)?;
    let mut input_scales_dev = DeviceBuffer::<u8>::zeroed(&stream, HiddenState::LEN / 16)?;
    let mut input_global_scales_dev = DeviceBuffer::<f32>::zeroed(&stream, GPT2_CONTEXT_LEN)?;
    let mut qkv_dev = DeviceBuffer::<f32>::zeroed(&stream, QkvActivation::LEN)?;
    let mut attention_log_sum_exp_dev =
        DeviceBuffer::<f32>::zeroed(&stream, AttentionLogSumExp::LEN)?;
    let mut tc_q_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let mut tc_k_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let mut tc_v_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let square = GPT2_N_HEAD * GPT2_CONTEXT_LEN * GPT2_CONTEXT_LEN;
    let mut tc_scores_dev = DeviceBuffer::<f32>::zeroed(&stream, square)?;
    let mut tc_probs_dev = DeviceBuffer::<f32>::zeroed(&stream, square)?;
    let mut tc_out_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let mut tc_chunk_states_dev = DeviceBuffer::<u16>::zeroed(&stream, HiddenState::LEN)?;

    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &qkv_identity_weight_bytes())?;
    let weight_scales_dev = filled_u8(&stream, QkvWeightShape::SCALE_LEN, E4M3_ONE)?;

    let bias_bytes_dev = filled_u8(&stream, QkvVectorShape::BYTE_LEN, 0)?;
    let bias_scales_dev = filled_u8(&stream, QkvVectorShape::SCALE_LEN, E4M3_ONE)?;
    let global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;

    let c_proj_weight_bytes_dev =
        DeviceBuffer::from_host(&stream, &c_proj_identity_weight_bytes())?;
    let c_proj_weight_scales_dev = filled_u8(&stream, ResidualWeightShape::SCALE_LEN, E4M3_ONE)?;

    let c_proj_bias_bytes_dev = filled_u8(&stream, HiddenVectorShape::BYTE_LEN, 0)?;
    let c_proj_bias_scales_dev = filled_u8(&stream, HiddenVectorShape::SCALE_LEN, E4M3_ONE)?;

    AttentionWeights::forward(AttentionForwardArgs {
        use_full_attention: true,
        module: &attention_module,
        tc_module: &tc_module,
        quant_module: &quant_module,
        input_nvfp4: HiddenStateNvfp4 {
            bytes: &mut input_bytes_dev,
            scales: &mut input_scales_dev,
            global_scales: &mut input_global_scales_dev,
        },
        tc_scratch: CausalAttentionTcScratch {
            q: &mut tc_q_dev,
            k: &mut tc_k_dev,
            v: &mut tc_v_dev,
            scores: &mut tc_scores_dev,
            probs: &mut tc_probs_dev,
            compact_out: &mut tc_out_dev,
            chunk_states: &mut tc_chunk_states_dev,
        },
        projections: AttentionProjectionTensors {
            qkv_weight: Nvfp4FourSixMmaWeightTensor::new(
                &weight_bytes_dev,
                &weight_scales_dev,
                &global_scale_dev,
            ),
            qkv_bias: Nvfp4DeviceTensor::new(&bias_bytes_dev, &bias_scales_dev, &global_scale_dev),
            c_proj_weight: Nvfp4FourSixMmaWeightTensor::new(
                &c_proj_weight_bytes_dev,
                &c_proj_weight_scales_dev,
                &global_scale_dev,
            ),
            c_proj_bias: Nvfp4DeviceTensor::new(
                &c_proj_bias_bytes_dev,
                &c_proj_bias_scales_dev,
                &global_scale_dev,
            ),
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
