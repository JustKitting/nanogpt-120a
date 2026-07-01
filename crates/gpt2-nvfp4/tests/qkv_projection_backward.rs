use std::error::Error;

use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionBackwardModules, AttentionProjectionTensors, AttentionQkvBackwardArgs, GPT2_N_EMBD,
    GPT2_QKV, HiddenState, qkv_projection_backward,
};
use rust_kernels_cuda::linear_backward::LinearBackwardModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::{Nvfp4DecodeModule, Nvfp4DeviceTensor};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::transpose::TransposeModule;

mod common;
#[path = "qkv_projection_backward/data.rs"]
mod data;
#[path = "qkv_projection_backward/scratch.rs"]
mod scratch;

use common::{assert_nonzero_finite, cuda_test_context};

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn qkv_projection_backward_runs_linear_ms_eden_path() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = cuda_test_context()?;
    let transpose = TransposeModule::from_module(ptx.clone())?;
    let decode = Nvfp4DecodeModule::from_module(ptx.clone())?;
    let linear = LinearBackwardModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;

    let qkv_input_bytes = DeviceBuffer::from_host(&stream, &data::qkv_input_bytes())?;
    let qkv_input_scales = DeviceBuffer::from_host(&stream, &data::hidden_scales())?;
    let qkv_input_globals = DeviceBuffer::from_host(&stream, &data::row_global_scales())?;
    let qkv_weight_bytes = DeviceBuffer::from_host(&stream, &data::qkv_weight_bytes())?;
    let qkv_weight_scales = DeviceBuffer::from_host(&stream, &data::qkv_weight_scales())?;
    let zero_bytes = DeviceBuffer::from_host(&stream, &data::zero_bytes())?;
    let one_scales = DeviceBuffer::from_host(&stream, &data::one_scales())?;
    let d_qkv = DeviceBuffer::from_host(&stream, &data::d_qkv_values())?;
    let dummy_f32 = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let dummy_u16 = DeviceBuffer::<u16>::zeroed(&stream, 1)?;
    let global_scale = DeviceBuffer::from_host(&stream, &[1.0_f32])?;

    let saved = data::saved_block(
        &qkv_input_bytes,
        &qkv_input_scales,
        &qkv_input_globals,
        &dummy_f32,
        &dummy_u16,
    );
    let projections = AttentionProjectionTensors {
        qkv_weight: Nvfp4FourSixMmaWeightTensor::new(
            &qkv_weight_bytes,
            &qkv_weight_scales,
            &global_scale,
        ),
        qkv_bias: Nvfp4DeviceTensor::new(&zero_bytes, &one_scales, &global_scale),
        c_proj_weight: Nvfp4FourSixMmaWeightTensor::new(&zero_bytes, &one_scales, &global_scale),
        c_proj_bias: Nvfp4DeviceTensor::new(&zero_bytes, &one_scales, &global_scale),
    };
    let mut scratch = scratch::QkvBackwardScratch::new(&stream)?;
    let mut d_ln_1_normalized = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let mut d_attn_qkv_weight = DeviceBuffer::<f32>::zeroed(&stream, GPT2_N_EMBD * GPT2_QKV)?;
    let mut d_attn_qkv_bias = DeviceBuffer::<f32>::zeroed(&stream, GPT2_QKV)?;

    qkv_projection_backward(AttentionQkvBackwardArgs {
        use_full_attention: false,
        stream: &stream,
        modules: AttentionBackwardModules {
            transpose: &transpose,
            decode: &decode,
            linear: &linear,
            quant: &quant,
        },
        saved,
        projections,
        d_qkv: &d_qkv,
        d_ln_1_normalized: &mut d_ln_1_normalized,
        d_attn_qkv_weight: &mut d_attn_qkv_weight,
        d_attn_qkv_bias: &mut d_attn_qkv_bias,
        scratch: scratch.as_attention_scratch(),
        seeds: data::seeds(),
    })?;

    assert_nonzero_finite(&d_ln_1_normalized.to_host_vec(&stream)?);
    assert_nonzero_finite(&d_attn_qkv_weight.to_host_vec(&stream)?);
    assert_nonzero_finite(&d_attn_qkv_bias.to_host_vec(&stream)?);
    Ok(())
}
