use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::LinearBackwardInputTranspose;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use super::types::{AttentionBackwardModules, AttentionLinearScratch};
use crate::backward::linear::{LinearBackwardCall, nvfp4_weight_t, run_linear_backward};

pub(super) struct AttentionLinearPass<'a, 'scratch, 'out> {
    pub e: &'a DeviceBuffer<f32>,
    pub saved_input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub scratch: AttentionLinearScratch<'scratch>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
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
    run_linear_backward(LinearBackwardCall {
        stream,
        module: modules.linear,
        quant: modules.quant,
        e: pass.e,
        weight_t: nvfp4_weight_t(pass.weight),
        input_t: LinearBackwardInputTranspose::RowwiseNvfp4(pass.saved_input),
        scratch: pass.scratch.linear,
        dinput: pass.dinput,
        dweight: pass.dweight,
        dbias: Some(pass.dbias),
        token_count: pass.row_count,
        input_dim: pass.input_dim,
        output_dim: pass.output_dim,
        sign_seed: pass.sign_seed,
        scale_seed: pass.scale_seed,
        precomputed_e_amax_chunks: None,
    })
}
