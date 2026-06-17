use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::layer_norm_backward::{
    LayerNormBackwardInputArgs, LayerNormBackwardModule, LayerNormBackwardParamArgs,
};

use crate::{GPT2_CONTEXT_LEN, GPT2_N_EMBD};
use crate::{LayerNormGrads, LayerNormSaved, LayerNormTensors};

pub struct Gpt2LayerNormBackwardInputArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub weights: LayerNormTensors<'a>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
}

pub struct Gpt2LayerNormBackwardParamArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub d_weight: &'out mut DeviceBuffer<f32>,
    pub d_bias: &'out mut DeviceBuffer<f32>,
}

pub struct Gpt2LayerNormBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub weights: LayerNormTensors<'a>,
    pub grads: LayerNormGrads<'out>,
}

pub fn layer_norm_backward_input(
    args: Gpt2LayerNormBackwardInputArgs<'_, '_>,
) -> Result<(), DriverError> {
    args.module.backward_input(LayerNormBackwardInputArgs {
        stream: args.stream,
        residual: args.saved.residual,
        d_normalized: args.d_normalized,
        mean: args.saved.mean,
        inv_std: args.saved.inv_std,
        weight: args.weights.weight,
        d_residual: args.d_residual,
        row_count: GPT2_CONTEXT_LEN as u32,
        embedding_dim: GPT2_N_EMBD as u32,
    })
}

pub fn layer_norm_backward_params(
    args: Gpt2LayerNormBackwardParamArgs<'_, '_>,
) -> Result<(), DriverError> {
    args.module.backward_params(LayerNormBackwardParamArgs {
        stream: args.stream,
        residual: args.saved.residual,
        d_normalized: args.d_normalized,
        mean: args.saved.mean,
        inv_std: args.saved.inv_std,
        d_weight: args.d_weight,
        d_bias: args.d_bias,
        row_count: GPT2_CONTEXT_LEN as u32,
        embedding_dim: GPT2_N_EMBD as u32,
    })
}

pub fn layer_norm_backward(args: Gpt2LayerNormBackwardArgs<'_, '_>) -> Result<(), DriverError> {
    let LayerNormGrads {
        d_residual,
        d_normalized,
        d_weight,
        d_bias,
    } = args.grads;

    layer_norm_backward_params(Gpt2LayerNormBackwardParamArgs {
        stream: args.stream,
        module: args.module,
        saved: args.saved,
        d_normalized,
        d_weight,
        d_bias,
    })?;
    layer_norm_backward_input(Gpt2LayerNormBackwardInputArgs {
        stream: args.stream,
        module: args.module,
        saved: args.saved,
        weights: args.weights,
        d_normalized,
        d_residual,
    })
}
