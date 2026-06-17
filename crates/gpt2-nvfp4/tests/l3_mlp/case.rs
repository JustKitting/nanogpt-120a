use std::error::Error;

use cuda_core::CudaContext;
use gpt2_nvfp4::{
    HiddenStateDevice, HiddenStateNvfp4, MlpActivationNvfp4, MlpProjectionTensors, MlpScratch,
    MlpWeights,
};
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use crate::assertions::{assert_down_projection_residual_add, assert_relu2_samples};
use crate::buffers::ScratchBuffers;
use crate::data::{normalized_input, residual_input};
use crate::runtime::{gpu_device_index, ptx_path};
use crate::weights::WeightBuffers;

pub fn run() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module = ctx.load_module_from_file(ptx_path().as_str())?;
    let mlp_module = MlpModule::from_module(module.clone())?;
    let quant_module = Nvfp4QuantModule::from_module(module)?;

    let normalized = normalized_input();
    let amax = vec![0.5_f32; gpt2_nvfp4::GPT2_CONTEXT_LEN];
    let residual_before = residual_input();
    let mut scratch = ScratchBuffers::new(&stream, &normalized, &amax, &residual_before)?;
    let weights = WeightBuffers::new(&stream)?;

    MlpWeights::forward(MlpWeights::input_from_attention(
        &mlp_module,
        &quant_module,
        MlpScratch {
            input_nvfp4: HiddenStateNvfp4 {
                bytes: &mut scratch.input_bytes,
                scales: &mut scratch.input_scales,
                global_scales: &mut scratch.input_global_scales,
            },
            activation_nvfp4: MlpActivationNvfp4 {
                bytes: &mut scratch.activation_bytes,
                scales: &mut scratch.activation_scales,
                global_scales: &mut scratch.activation_global_scales,
            },
            pre_activation: None,
            activation: &mut scratch.activation,
        },
        MlpProjectionTensors {
            up: weights.up_tensors(),
            down: weights.down_tensors(),
        },
        HiddenStateDevice {
            stream: &stream,
            residual: &mut scratch.residual,
            normalized: &mut scratch.normalized,
            normalized_amax: &mut scratch.amax,
        },
    ))?;

    let activation = scratch.activation.to_host_vec(&stream)?;
    let residual_after = scratch.residual.to_host_vec(&stream)?;
    assert_relu2_samples(&activation);
    assert_down_projection_residual_add(&residual_before, &residual_after);
    Ok(())
}
