use std::error::Error;
use std::path::PathBuf;

use cuda_core::{CudaContext, DeviceBuffer};
use gpt2_nvfp4::{
    AttentionInputNvfp4, AttentionWeights, GPT2_CONTEXT_LEN, GPT2_N_EMBD, GPT2_QKV, HiddenState,
    HiddenStateDevice, Nvfp4Shape, QkvActivation, QkvVectorShape, QkvWeightShape,
};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

const E2M1_ONE_PAIR: u8 = 0x22;
const E4M3_ONE: u8 = 0x38;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn attention_forward_quantizes_projects_and_applies_causal_attention() -> Result<(), Box<dyn Error>>
{
    let ctx = CudaContext::new(gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module = ctx.load_module_from_file(ptx_path().as_str())?;
    let attention_module = AttentionModule::from_module(module.clone())?;
    let quant_module = Nvfp4QuantModule::from_module(module)?;

    let (hidden, amax) = hidden_input();
    let mut residual_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let mut hidden_dev = DeviceBuffer::from_host(&stream, &hidden)?;
    let mut amax_dev = DeviceBuffer::from_host(&stream, &amax)?;
    let mut input_bytes_dev = DeviceBuffer::<u8>::zeroed(&stream, HiddenState::LEN / 2)?;
    let mut input_scales_dev = DeviceBuffer::<u8>::zeroed(&stream, HiddenState::LEN / 16)?;
    let mut input_global_scales_dev = DeviceBuffer::<f32>::zeroed(&stream, GPT2_CONTEXT_LEN)?;
    let mut qkv_dev = DeviceBuffer::<f32>::zeroed(&stream, QkvActivation::LEN)?;

    let weight_bytes = qkv_value_weight_bytes();
    let weight_scales = vec![E4M3_ONE; QkvWeightShape::SCALE_LEN];
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &weight_bytes)?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &weight_scales)?;

    let bias_bytes = vec![0_u8; QkvVectorShape::BYTE_LEN];
    let bias_scales = vec![E4M3_ONE; QkvVectorShape::SCALE_LEN];
    let bias_bytes_dev = DeviceBuffer::from_host(&stream, &bias_bytes)?;
    let bias_scales_dev = DeviceBuffer::from_host(&stream, &bias_scales)?;

    AttentionWeights::forward(AttentionWeights::input_from_embeddings(
        &attention_module,
        &quant_module,
        AttentionInputNvfp4 {
            bytes: &mut input_bytes_dev,
            scales: &mut input_scales_dev,
            global_scales: &mut input_global_scales_dev,
        },
        Nvfp4FourSixMmaWeightTensor {
            bytes: &weight_bytes_dev,
            scales: &weight_scales_dev,
            global_scale: 1.0,
        },
        Nvfp4DeviceTensor {
            bytes: &bias_bytes_dev,
            scales: &bias_scales_dev,
            global_scale: 1.0,
        },
        &mut qkv_dev,
        HiddenStateDevice {
            stream: &stream,
            residual: &mut residual_dev,
            normalized: &mut hidden_dev,
            normalized_amax: &mut amax_dev,
        },
    ))?;

    let qkv = qkv_dev.to_host_vec(&stream)?;
    let out = hidden_dev.to_host_vec(&stream)?;
    let output_amax = amax_dev.to_host_vec(&stream)?;
    assert_qk_zero_and_v_nonzero(&qkv);
    assert_prefix_average(&qkv, &out);
    assert_output_amax(&out, &output_amax);
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

fn qkv_value_weight_bytes() -> Vec<u8> {
    let mut bytes = vec![0_u8; QkvWeightShape::BYTE_LEN];
    for col in (2 * GPT2_N_EMBD)..GPT2_QKV {
        let byte_base = col * GPT2_N_EMBD / 2;
        bytes[byte_base..byte_base + GPT2_N_EMBD / 2].fill(E2M1_ONE_PAIR);
    }
    bytes
}

fn assert_qk_zero_and_v_nonzero(qkv: &[f32]) {
    let mut v_nonzero = false;
    for row in 0..GPT2_CONTEXT_LEN {
        let row_base = row * GPT2_QKV;
        for value in &qkv[row_base..row_base + 2 * GPT2_N_EMBD] {
            assert!(value.abs() <= 1.0e-5, "q_or_k={value:.8e}");
        }
        v_nonzero |= qkv[row_base + 2 * GPT2_N_EMBD..row_base + GPT2_QKV]
            .iter()
            .any(|value| value.abs() > 1.0e-5);
    }
    assert!(v_nonzero);
}

fn assert_prefix_average(qkv: &[f32], out: &[f32]) {
    let mut prefix = vec![0.0_f32; GPT2_N_EMBD];
    for row in 0..GPT2_CONTEXT_LEN {
        let v_base = row * GPT2_QKV + 2 * GPT2_N_EMBD;
        let out_base = row * GPT2_N_EMBD;
        for col in 0..GPT2_N_EMBD {
            prefix[col] += qkv[v_base + col];
            let expected = prefix[col] / (row + 1) as f32;
            let actual = out[out_base + col];
            let error = (actual - expected).abs();
            let tolerance = expected.abs().max(1.0) * 1.0e-5;
            assert!(
                error <= tolerance,
                "row={row} col={col} actual={actual:.8e} expected={expected:.8e} error={error:.8e} tolerance={tolerance:.8e}"
            );
        }
    }
}

fn assert_output_amax(out: &[f32], output_amax: &[f32]) {
    for (row, actual) in output_amax.iter().copied().enumerate() {
        let row_base = row * GPT2_N_EMBD;
        let expected = out[row_base..row_base + GPT2_N_EMBD]
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        let error = (actual - expected).abs();
        let tolerance = expected.abs().max(1.0) * 1.0e-5;
        assert!(
            error <= tolerance,
            "row={row} actual_amax={actual:.8e} expected_amax={expected:.8e} error={error:.8e} tolerance={tolerance:.8e}"
        );
    }
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
