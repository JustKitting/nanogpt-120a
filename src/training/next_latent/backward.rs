use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{Gpt2Rng, GPT2_N_EMBD, NEXTLAT_HIDDEN};
use rust_kernels_cuda::layer_norm_backward::LayerNormBackwardModule;
use rust_kernels_cuda::linear_backward::LinearBackwardModule;
use rust_kernels_cuda::next_latent::{
    NextLatConcatBackwardArgs, NextLatGeluBackwardArgs, NextLatModule,
};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use super::backward_linear::{
    input_projection_backward, output_projection_backward, transition_backward,
};
use super::backward_norm::layer_norm_backward;
use super::buffers::NextLatBuffers;
use super::grads::NextLatGradBuffers;
use super::scratch::NextLatScratchBuffers;
use crate::upload::UploadedNextLat;

pub struct NextLatBackwardArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub next_latent: &'a NextLatModule,
    pub linear: &'a LinearBackwardModule,
    pub quant: &'a Nvfp4QuantModule,
    pub layer_norm: &'a LayerNormBackwardModule,
    pub weights: &'a UploadedNextLat,
    pub forward: &'a NextLatBuffers,
    pub grads: &'out mut NextLatGradBuffers,
    pub scratch: &'scratch mut NextLatScratchBuffers,
    pub row_count: u32,
    pub seeds: NextLatBackwardSeeds,
}

#[derive(Clone, Copy)]
pub struct NextLatBackwardSeeds {
    pub output_sign: u32,
    pub output_scale: u32,
    pub transition_sign: u32,
    pub transition_scale: u32,
    pub input_sign: u32,
    pub input_scale: u32,
}

impl NextLatBackwardSeeds {
    pub fn from_rng(rng: &mut Gpt2Rng) -> Self {
        Self {
            output_sign: rng.next_u32(),
            output_scale: rng.next_u32(),
            transition_sign: rng.next_u32(),
            transition_scale: rng.next_u32(),
            input_sign: rng.next_u32(),
            input_scale: rng.next_u32(),
        }
    }
}

pub fn backward(mut args: NextLatBackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    output_projection_backward(&mut args)?;
    gelu_backward(
        args.next_latent,
        args.stream,
        &args.forward.pre2,
        &args.grads.d_act2,
        &mut args.grads.d_pre2,
        args.row_count * NEXTLAT_HIDDEN as u32,
    )?;
    transition_backward(&mut args)?;
    gelu_backward(
        args.next_latent,
        args.stream,
        &args.forward.pre1,
        &args.grads.d_act1,
        &mut args.grads.d_pre1,
        args.row_count * NEXTLAT_HIDDEN as u32,
    )?;
    input_projection_backward(&mut args)?;
    layer_norm_backward(&mut args)?;
    args.next_latent.concat_backward(NextLatConcatBackwardArgs {
        stream: args.stream,
        d_concat: &args.grads.d_concat,
        d_predicted: &args.forward.d_predicted,
        d_next_token_embeddings: &mut args.grads.d_next_token_embeddings,
        d_current_states: &mut args.grads.d_current_states,
        row_count: args.row_count,
        embedding_dim: GPT2_N_EMBD as u32,
    })
}

fn gelu_backward(
    module: &NextLatModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    d_out: &DeviceBuffer<f32>,
    d_input: &mut DeviceBuffer<f32>,
    len: u32,
) -> Result<(), DriverError> {
    module.gelu_backward(NextLatGeluBackwardArgs {
        stream,
        input,
        d_out,
        d_input,
        len,
    })
}
