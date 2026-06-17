use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::LinearBackwardMsEdenArgs;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use super::transforms::{decode_rowwise_t, decode_weight_t, transpose_f32};
use super::types::{AttentionBackwardModules, AttentionLinearScratch};
use crate::GPT2_CONTEXT_LEN;

pub(super) struct AttentionLinearPass<'a, 'scratch, 'out> {
    pub e: &'a DeviceBuffer<f32>,
    pub saved_input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub scratch: AttentionLinearScratch<'scratch>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: &'out mut DeviceBuffer<f32>,
    pub input_dim: u32,
    pub output_dim: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub(super) fn run_attention_linear_pass(
    modules: &AttentionBackwardModules<'_>,
    stream: &CudaStream,
    pass: AttentionLinearPass<'_, '_, '_>,
) -> Result<(), DriverError> {
    decode_weight_t(
        modules.decode,
        stream,
        pass.weight,
        pass.scratch.weight_t,
        pass.output_dim as usize,
        pass.input_dim as usize,
    )?;
    transpose_f32(
        modules.transpose,
        stream,
        pass.e,
        pass.scratch.error_t,
        GPT2_CONTEXT_LEN,
        pass.output_dim as usize,
    )?;
    decode_rowwise_t(
        modules.decode,
        stream,
        pass.saved_input,
        pass.scratch.input_t,
        GPT2_CONTEXT_LEN,
        pass.input_dim as usize,
    )?;

    modules.linear.backward_ms_eden(LinearBackwardMsEdenArgs {
        stream,
        quant_module: modules.quant,
        e: pass.e,
        weight_t: pass.scratch.weight_t,
        e_t: pass.scratch.error_t,
        input_t: pass.scratch.input_t,
        scratch: pass.scratch.linear,
        dinput: pass.dinput,
        dweight: pass.dweight,
        dbias: Some(pass.dbias),
        token_count: GPT2_CONTEXT_LEN as u32,
        input_dim: pass.input_dim,
        output_dim: pass.output_dim,
        sign_seed: pass.sign_seed,
        scale_seed: pass.scale_seed,
    })
}
