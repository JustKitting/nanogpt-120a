use cuda_core::CudaStream;
use gpt2_nvfp4::Gpt2Rng;
use rust_kernels_cuda::layer_norm_backward::LayerNormBackwardModule;
use rust_kernels_cuda::linear_backward::LinearBackwardModule;
use rust_kernels_cuda::next_latent::NextLatModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use super::super::buffers::NextLatBuffers;
use super::super::grads::NextLatGradBuffers;
use super::super::scratch::NextLatScratchBuffers;
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
