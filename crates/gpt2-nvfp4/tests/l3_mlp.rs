use std::error::Error;
use std::path::PathBuf;

use cuda_core::{CudaContext, DeviceBuffer};
use gpt2_nvfp4::{
    GPT2_CONTEXT_LEN, GPT2_MLP, GPT2_N_EMBD, HiddenState, HiddenStateDevice, HiddenStateNvfp4,
    HiddenVectorShape, MlpActivation, MlpActivationNvfp4, MlpDownTensors, MlpDownWeightShape,
    MlpProjectionTensors, MlpScratch, MlpUpTensors, MlpUpWeightShape, MlpVectorShape, MlpWeights,
    Nvfp4Shape,
};
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

const E4M3_ONE: u8 = 0x38;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn mlp_forward_projects_relu2_downprojects_and_residual_adds() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module = ctx.load_module_from_file(ptx_path().as_str())?;
    let mlp_module = MlpModule::from_module(module.clone())?;
    let quant_module = Nvfp4QuantModule::from_module(module)?;

    let normalized = normalized_input();
    let amax = vec![0.5_f32; GPT2_CONTEXT_LEN];
    let residual_before = residual_input();
    let mut residual_dev = DeviceBuffer::from_host(&stream, &residual_before)?;
    let mut normalized_dev = DeviceBuffer::from_host(&stream, &normalized)?;
    let mut amax_dev = DeviceBuffer::from_host(&stream, &amax)?;
    let mut input_bytes_dev = DeviceBuffer::<u8>::zeroed(&stream, HiddenState::LEN / 2)?;
    let mut input_scales_dev = DeviceBuffer::<u8>::zeroed(&stream, HiddenState::LEN / 16)?;
    let mut input_global_scales_dev = DeviceBuffer::<f32>::zeroed(&stream, GPT2_CONTEXT_LEN)?;
    let mut activation_dev = DeviceBuffer::<f32>::zeroed(&stream, MlpActivation::LEN)?;
    let mut activation_bytes_dev = DeviceBuffer::<u8>::zeroed(&stream, MlpActivation::LEN / 2)?;
    let mut activation_scales_dev = DeviceBuffer::<u8>::zeroed(&stream, MlpActivation::LEN / 16)?;
    let mut activation_global_scales_dev = DeviceBuffer::<f32>::zeroed(&stream, GPT2_CONTEXT_LEN)?;

    let weight_bytes = mlp_up_repeat_weight_bytes();
    let weight_scales = vec![E4M3_ONE; MlpUpWeightShape::SCALE_LEN];
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &weight_bytes)?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &weight_scales)?;

    let bias_bytes = vec![0_u8; MlpVectorShape::BYTE_LEN];
    let bias_scales = vec![E4M3_ONE; MlpVectorShape::SCALE_LEN];
    let bias_bytes_dev = DeviceBuffer::from_host(&stream, &bias_bytes)?;
    let bias_scales_dev = DeviceBuffer::from_host(&stream, &bias_scales)?;

    let down_weight_bytes = mlp_down_identity_weight_bytes();
    let down_weight_scales = vec![E4M3_ONE; MlpDownWeightShape::SCALE_LEN];
    let down_weight_bytes_dev = DeviceBuffer::from_host(&stream, &down_weight_bytes)?;
    let down_weight_scales_dev = DeviceBuffer::from_host(&stream, &down_weight_scales)?;

    let down_bias_bytes = vec![0_u8; HiddenVectorShape::BYTE_LEN];
    let down_bias_scales = vec![E4M3_ONE; HiddenVectorShape::SCALE_LEN];
    let down_bias_bytes_dev = DeviceBuffer::from_host(&stream, &down_bias_bytes)?;
    let down_bias_scales_dev = DeviceBuffer::from_host(&stream, &down_bias_scales)?;

    MlpWeights::forward(MlpWeights::input_from_attention(
        &mlp_module,
        &quant_module,
        MlpScratch {
            input_nvfp4: HiddenStateNvfp4 {
                bytes: &mut input_bytes_dev,
                scales: &mut input_scales_dev,
                global_scales: &mut input_global_scales_dev,
            },
            activation_nvfp4: MlpActivationNvfp4 {
                bytes: &mut activation_bytes_dev,
                scales: &mut activation_scales_dev,
                global_scales: &mut activation_global_scales_dev,
            },
            activation: &mut activation_dev,
        },
        MlpProjectionTensors {
            up: MlpUpTensors {
                weight: Nvfp4FourSixMmaWeightTensor {
                    bytes: &weight_bytes_dev,
                    scales: &weight_scales_dev,
                    global_scale: 1.0,
                },
                bias: Nvfp4DeviceTensor {
                    bytes: &bias_bytes_dev,
                    scales: &bias_scales_dev,
                    global_scale: 1.0,
                },
            },
            down: MlpDownTensors {
                weight: Nvfp4FourSixMmaWeightTensor {
                    bytes: &down_weight_bytes_dev,
                    scales: &down_weight_scales_dev,
                    global_scale: 1.0,
                },
                bias: Nvfp4DeviceTensor {
                    bytes: &down_bias_bytes_dev,
                    scales: &down_bias_scales_dev,
                    global_scale: 1.0,
                },
            },
        },
        HiddenStateDevice {
            stream: &stream,
            residual: &mut residual_dev,
            normalized: &mut normalized_dev,
            normalized_amax: &mut amax_dev,
        },
    ))?;

    let activation = activation_dev.to_host_vec(&stream)?;
    let residual_after = residual_dev.to_host_vec(&stream)?;
    assert_relu2_samples(&activation);
    assert_down_projection_residual_add(&residual_before, &residual_after);
    Ok(())
}

fn normalized_input() -> Vec<f32> {
    let mut normalized = vec![0.0_f32; HiddenState::LEN];
    for row in 0..GPT2_CONTEXT_LEN {
        let row_base = row * GPT2_N_EMBD;
        normalized[row_base..row_base + GPT2_N_EMBD / 2].fill(0.5);
        normalized[row_base + GPT2_N_EMBD / 2..row_base + GPT2_N_EMBD].fill(-0.5);
    }
    normalized
}

fn residual_input() -> Vec<f32> {
    let mut residual = vec![0.0_f32; HiddenState::LEN];
    for row in 0..GPT2_CONTEXT_LEN {
        let row_base = row * GPT2_N_EMBD;
        for col in 0..GPT2_N_EMBD {
            residual[row_base + col] = 0.125 + row as f32 * 0.000_244_140_62 + col as f32 * 1.0e-6;
        }
    }
    residual
}

fn mlp_up_repeat_weight_bytes() -> Vec<u8> {
    let mut bytes = vec![0_u8; MlpUpWeightShape::BYTE_LEN];
    for col in 0..GPT2_MLP {
        set_e2m1_one(&mut bytes, col * GPT2_N_EMBD + col % GPT2_N_EMBD);
    }
    bytes
}

fn mlp_down_identity_weight_bytes() -> Vec<u8> {
    let mut bytes = vec![0_u8; MlpDownWeightShape::BYTE_LEN];
    for col in 0..GPT2_N_EMBD {
        set_e2m1_one(&mut bytes, col * GPT2_MLP + col);
    }
    bytes
}

fn set_e2m1_one(bytes: &mut [u8], element: usize) {
    let byte = &mut bytes[element / 2];
    if element & 1 == 0 {
        *byte = (*byte & 0xf0) | 0x2;
    } else {
        *byte = (*byte & 0x0f) | 0x20;
    }
}

fn assert_relu2_samples(activation: &[f32]) {
    for row in [0, 1, 17, GPT2_CONTEXT_LEN - 1] {
        assert_positive_relu2(activation, row, 0);
        assert_positive_relu2(activation, row, 37);
        assert_positive_relu2(activation, row, GPT2_N_EMBD + 11);
        assert_zero_relu2(activation, row, GPT2_N_EMBD / 2);
        assert_zero_relu2(activation, row, GPT2_N_EMBD + GPT2_N_EMBD / 2 + 5);
    }
}

fn assert_positive_relu2(activation: &[f32], row: usize, col: usize) {
    let actual = activation[row * GPT2_MLP + col];
    let expected = 0.25_f32;
    let error = (actual - expected).abs();
    assert!(
        error <= 5.0e-2,
        "row={row} col={col} actual={actual:.8e} expected={expected:.8e} error={error:.8e}"
    );
}

fn assert_zero_relu2(activation: &[f32], row: usize, col: usize) {
    let actual = activation[row * GPT2_MLP + col];
    assert!(
        actual.abs() <= 1.0e-6,
        "row={row} col={col} actual={actual:.8e}"
    );
}

fn assert_down_projection_residual_add(residual_before: &[f32], residual_after: &[f32]) {
    for row in [0, 1, 17, GPT2_CONTEXT_LEN - 1] {
        assert_residual_delta(residual_before, residual_after, row, 0, 0.25);
        assert_residual_delta(residual_before, residual_after, row, 37, 0.25);
        assert_residual_delta(residual_before, residual_after, row, GPT2_N_EMBD / 2, 0.0);
        assert_residual_delta(residual_before, residual_after, row, GPT2_N_EMBD - 1, 0.0);
    }
}

fn assert_residual_delta(
    residual_before: &[f32],
    residual_after: &[f32],
    row: usize,
    col: usize,
    expected_delta: f32,
) {
    let index = row * GPT2_N_EMBD + col;
    let actual_delta = residual_after[index] - residual_before[index];
    let error = (actual_delta - expected_delta).abs();
    assert!(
        error <= 5.0e-2,
        "row={row} col={col} actual_delta={actual_delta:.8e} expected_delta={expected_delta:.8e} error={error:.8e}"
    );
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
