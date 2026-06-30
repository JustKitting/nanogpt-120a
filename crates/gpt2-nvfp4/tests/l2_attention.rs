use std::error::Error;

use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionForwardArgs, AttentionLogSumExp, AttentionProjectionTensors, AttentionWeights,
    GPT2_CONTEXT_LEN, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_QKV, HiddenState, HiddenStateDevice,
    HiddenStateNvfp4, HiddenVectorShape, Nvfp4Shape, QkvActivation, QkvVectorShape, QkvWeightShape,
    ResidualWeightShape,
};
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionTcScratch};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

mod common;
#[path = "common/f16.rs"]
mod f16_common;
#[path = "common/nvfp4.rs"]
mod nvfp4_common;

use common::cuda_test_context;
use f16_common::tc_f16;
use nvfp4_common::repeating_identity_bytes;

const E4M3_ONE: u8 = 0x38;
const HEAD_DIM: usize = GPT2_N_EMBD / GPT2_N_HEAD;
const ATTENTION_TOLERANCE: f32 = 1.0e-6;
const RESIDUAL_TOLERANCE: f32 = 1.0e-7;

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

    let weight_bytes = qkv_identity_weight_bytes();
    let weight_scales = vec![E4M3_ONE; QkvWeightShape::SCALE_LEN];
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &weight_bytes)?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &weight_scales)?;

    let bias_bytes = vec![0_u8; QkvVectorShape::BYTE_LEN];
    let bias_scales = vec![E4M3_ONE; QkvVectorShape::SCALE_LEN];
    let bias_bytes_dev = DeviceBuffer::from_host(&stream, &bias_bytes)?;
    let bias_scales_dev = DeviceBuffer::from_host(&stream, &bias_scales)?;
    let global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;

    let c_proj_weight_bytes = c_proj_identity_weight_bytes();
    let c_proj_weight_scales = vec![E4M3_ONE; ResidualWeightShape::SCALE_LEN];
    let c_proj_weight_bytes_dev = DeviceBuffer::from_host(&stream, &c_proj_weight_bytes)?;
    let c_proj_weight_scales_dev = DeviceBuffer::from_host(&stream, &c_proj_weight_scales)?;

    let c_proj_bias_bytes = vec![0_u8; HiddenVectorShape::BYTE_LEN];
    let c_proj_bias_scales = vec![E4M3_ONE; HiddenVectorShape::SCALE_LEN];
    let c_proj_bias_bytes_dev = DeviceBuffer::from_host(&stream, &c_proj_bias_bytes)?;
    let c_proj_bias_scales_dev = DeviceBuffer::from_host(&stream, &c_proj_bias_scales)?;

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
            qkv_weight: Nvfp4FourSixMmaWeightTensor {
                bytes: &weight_bytes_dev,
                scales: &weight_scales_dev,
                global_scale: &global_scale_dev,
            },
            qkv_bias: Nvfp4DeviceTensor {
                bytes: &bias_bytes_dev,
                scales: &bias_scales_dev,
                global_scale: &global_scale_dev,
            },
            c_proj_weight: Nvfp4FourSixMmaWeightTensor {
                bytes: &c_proj_weight_bytes_dev,
                scales: &c_proj_weight_scales_dev,
                global_scale: &global_scale_dev,
            },
            c_proj_bias: Nvfp4DeviceTensor {
                bytes: &c_proj_bias_bytes_dev,
                scales: &c_proj_bias_scales_dev,
                global_scale: &global_scale_dev,
            },
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

fn hidden_input() -> (Vec<f32>, Vec<f32>) {
    let mut hidden = vec![0.0_f32; HiddenState::LEN];
    let mut amax = vec![0.0_f32; GPT2_CONTEXT_LEN];
    for row in 0..GPT2_CONTEXT_LEN {
        let value = 0.125 + (row % 7) as f32 * 0.0625;
        hidden[row * GPT2_N_EMBD..(row + 1) * GPT2_N_EMBD].fill(value);
        amax[row] = value;
    }
    (hidden, amax)
}

fn residual_input() -> Vec<f32> {
    let mut residual = vec![0.0_f32; HiddenState::LEN];
    for row in 0..GPT2_CONTEXT_LEN {
        residual[row * GPT2_N_EMBD..(row + 1) * GPT2_N_EMBD]
            .fill(0.25 + row as f32 * 0.000_976_562_5);
    }
    residual
}

fn qkv_identity_weight_bytes() -> Vec<u8> {
    repeating_identity_bytes(QkvWeightShape::BYTE_LEN, GPT2_QKV, GPT2_N_EMBD)
}

fn c_proj_identity_weight_bytes() -> Vec<u8> {
    repeating_identity_bytes(ResidualWeightShape::BYTE_LEN, GPT2_N_EMBD, GPT2_N_EMBD)
}

fn assert_qkv_nonzero(qkv: &[f32]) {
    let q_nonzero = qkv
        .iter()
        .take(GPT2_N_EMBD)
        .any(|value| value.abs() > 1.0e-7);
    let k_nonzero = qkv[GPT2_N_EMBD..2 * GPT2_N_EMBD]
        .iter()
        .any(|value| value.abs() > 1.0e-7);
    let v_nonzero = qkv[2 * GPT2_N_EMBD..GPT2_QKV]
        .iter()
        .any(|value| value.abs() > 1.0e-7);
    assert!(q_nonzero && k_nonzero && v_nonzero);
}

fn assert_attention_log_sum_exp(log_sum_exp: &[f32]) {
    assert!(log_sum_exp.iter().all(|value| value.is_finite()));
    assert!(log_sum_exp.iter().any(|value| value.abs() > 1.0e-7));
}

fn assert_attention_matches(qkv: &[f32], out: &[f32]) {
    for row in [0, 1, 2, 17, 128, GPT2_CONTEXT_LEN - 1] {
        for head in [0, GPT2_N_HEAD / 2, GPT2_N_HEAD - 1] {
            let scores = attention_scores(qkv, row, head);
            let score_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = scores
                .iter()
                .map(|score| (*score - score_max).exp())
                .sum::<f32>();

            for dim in [0, 1, HEAD_DIM / 2, HEAD_DIM - 1] {
                let mut expected = 0.0;
                for (key, score) in scores.iter().copied().enumerate() {
                    let weight = tc_f16((score - score_max).exp() / denom);
                    let value = tc_f16(qkv_value(qkv, key, head, dim, 2 * GPT2_N_EMBD));
                    expected += weight * value;
                }

                let col = head * HEAD_DIM + dim;
                let actual = out[row * GPT2_N_EMBD + col];
                let error = (actual - expected).abs();
                let tolerance = expected.abs().max(1.0) * ATTENTION_TOLERANCE;
                assert!(
                    error <= tolerance,
                    "row={row} head={head} dim={dim} actual={actual:.8e} expected={expected:.8e} error={error:.8e} tolerance={tolerance:.8e}"
                );
            }
        }
    }
}

fn attention_scores(qkv: &[f32], query: usize, head: usize) -> Vec<f32> {
    let mut scores = Vec::with_capacity(query + 1);
    for key in 0..=query {
        let mut dot = 0.0;
        for dim in 0..HEAD_DIM {
            let q = tc_f16(qkv_value(qkv, query, head, dim, 0));
            let k = tc_f16(qkv_value(qkv, key, head, dim, GPT2_N_EMBD));
            dot += q * k;
        }
        scores.push(dot / (HEAD_DIM as f32).sqrt());
    }
    scores
}

fn qkv_value(qkv: &[f32], token: usize, head: usize, dim: usize, offset: usize) -> f32 {
    qkv[token * GPT2_QKV + offset + head * HEAD_DIM + dim]
}

fn assert_output_amax(out: &[f32], output_amax: &[f32]) {
    for (row, actual) in output_amax.iter().copied().enumerate() {
        let row_base = row * GPT2_N_EMBD;
        let expected = out[row_base..row_base + GPT2_N_EMBD]
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        let error = (actual - expected).abs();
        let tolerance = expected.abs().max(1.0) * 1.0e-7;
        assert!(
            error <= tolerance,
            "row={row} actual_amax={actual:.8e} expected_amax={expected:.8e} error={error:.8e} tolerance={tolerance:.8e}"
        );
    }
}

fn assert_c_proj_residual_add(
    residual_before: &[f32],
    attention_out: &[f32],
    residual_after: &[f32],
) {
    for index in 0..HiddenState::LEN {
        let expected = residual_before[index] + attention_out[index];
        let actual = residual_after[index];
        let error = (actual - expected).abs();
        let tolerance = expected.abs().max(1.0) * RESIDUAL_TOLERANCE;
        assert!(
            error <= tolerance,
            "index={index} actual={actual:.8e} expected={expected:.8e} error={error:.8e} tolerance={tolerance:.8e}"
        );
    }
}
