use cuda_core::CudaStream;
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::layer_norm_backward::LayerNormBackwardModule;
use rust_kernels_cuda::residual::ResidualBackwardModule;

use crate::Gpt2Rng;
use crate::LayerNormTensors;
use crate::backward::{
    AttentionBackwardModules, AttentionBackwardSeeds, AttentionCProjScratch, AttentionCoreScratch,
    AttentionQkvScratch,
};
use crate::types::{AttentionProjectionTensors, BlockBackwardGrads, BlockForwardSaved};

pub struct BlockAttentionBackwardModules<'a> {
    pub residual: &'a ResidualBackwardModule,
    pub layer_norm: &'a LayerNormBackwardModule,
    pub attention: &'a AttentionModule,
    pub linear: AttentionBackwardModules<'a>,
}

pub struct BlockAttentionBackwardScratch<'scratch> {
    pub c_proj: AttentionCProjScratch<'scratch>,
    pub core: AttentionCoreScratch<'scratch>,
    pub qkv: AttentionQkvScratch<'scratch>,
}

pub struct BlockAttentionBackwardSeeds {
    pub c_proj: AttentionBackwardSeeds,
    pub qkv: AttentionBackwardSeeds,
}

impl BlockAttentionBackwardSeeds {
    pub fn from_rng(rng: &mut Gpt2Rng) -> Self {
        Self {
            c_proj: AttentionBackwardSeeds::from_rng(rng),
            qkv: AttentionBackwardSeeds::from_rng(rng),
        }
    }
}

pub struct BlockAttentionBackwardArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub modules: BlockAttentionBackwardModules<'a>,
    pub saved: BlockForwardSaved<'a>,
    pub ln_1: LayerNormTensors<'a>,
    pub projections: AttentionProjectionTensors<'a>,
    pub grads: BlockBackwardGrads<'out>,
    pub scratch: BlockAttentionBackwardScratch<'scratch>,
    pub seeds: BlockAttentionBackwardSeeds,
}
