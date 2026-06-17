use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::layer_norm_backward::{LayerNormBackwardInputArgs, LayerNormBackwardModule};

use crate::{GPT2_CONTEXT_LEN, GPT2_N_EMBD};
use crate::{LayerNormSaved, LayerNormTensors};

pub struct Gpt2LayerNormBackwardInputArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a LayerNormBackwardModule,
    pub saved: LayerNormSaved<'a>,
    pub weights: LayerNormTensors<'a>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
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
