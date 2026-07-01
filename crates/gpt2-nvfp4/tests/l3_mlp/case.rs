use gpt2_nvfp4::{
    HiddenStateDevice, MlpForwardArgs, MlpProjectionTensors, MlpScratch, MlpWeights,
    GPT2_CONTEXT_LEN,
};
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use crate::assertions::{assert_down_projection_residual_add, assert_relu2_samples};
use crate::buffers::ScratchBuffers;
use crate::common::cuda_test_context;
use crate::data::{normalized_input, residual_input};
use crate::upload_common::TestResult;
use crate::weights::WeightBuffers;

pub fn run() -> TestResult {
    let (_, stream, module) = cuda_test_context()?;
    let mlp_module = MlpModule::from_module(module.clone())?;
    let quant_module = Nvfp4QuantModule::from_module(module)?;

    let normalized = normalized_input();
    let amax = vec![0.5_f32; gpt2_nvfp4::GPT2_CONTEXT_LEN];
    let residual_before = residual_input();
    let mut scratch = ScratchBuffers::new(&stream, &normalized, &amax, &residual_before)?;
    let weights = WeightBuffers::new(&stream)?;

    MlpWeights::forward(MlpForwardArgs {
        module: &mlp_module,
        quant_module: &quant_module,
        scratch: MlpScratch {
            input_nvfp4: scratch.input_nvfp4.args(),
            activation_nvfp4: scratch.activation_nvfp4.args(),
            pre_activation: &mut scratch.pre_activation,
            activation: &mut scratch.activation,
        },
        projections: MlpProjectionTensors {
            up: weights.up_tensors(),
            down: weights.down_tensors(),
        },
        hidden: HiddenStateDevice {
            stream: &stream,
            batch_size: 1,
            seq_len: GPT2_CONTEXT_LEN as u32,
            row_count: GPT2_CONTEXT_LEN as u32,
            residual: &mut scratch.residual,
            normalized: &mut scratch.normalized,
            normalized_amax: &mut scratch.amax,
            mean: &mut scratch.mean,
            inv_std: &mut scratch.inv_std,
        },
        tape: None,
    })?;

    let activation = scratch.activation.to_host_vec(&stream)?;
    let residual_after = scratch.residual.to_host_vec(&stream)?;
    assert_relu2_samples(&activation);
    assert_down_projection_residual_add(&residual_before, &residual_after);
    Ok(())
}
