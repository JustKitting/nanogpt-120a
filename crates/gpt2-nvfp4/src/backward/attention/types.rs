use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::linear_backward::{LinearBackwardModule, LinearBackwardMsEdenScratch};
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::transpose::TransposeModule;

use crate::Gpt2Rng;
use crate::types::{AttentionProjectionTensors, BlockForwardSaved};

pub struct AttentionBackwardModules<'a> {
    pub transpose: &'a TransposeModule,
    pub decode: &'a Nvfp4DecodeModule,
    pub linear: &'a LinearBackwardModule,
    pub quant: &'a Nvfp4QuantModule,
}

pub struct AttentionCProjScratch<'scratch> {
    pub error_t: &'scratch mut DeviceBuffer<f32>,
    pub weight_t: &'scratch mut DeviceBuffer<f32>,
    pub input_t: &'scratch mut DeviceBuffer<f32>,
    pub linear: LinearBackwardMsEdenScratch<'scratch>,
}

pub struct AttentionCoreScratch<'scratch> {
    pub softmax_d: &'scratch mut DeviceBuffer<f32>,
}

pub struct AttentionBackwardSeeds {
    pub(crate) sign: u32,
    pub(crate) scale: u32,
}

impl AttentionBackwardSeeds {
    pub fn from_rng(rng: &mut Gpt2Rng) -> Self {
        Self {
            sign: rng.next_u32(),
            scale: rng.next_u32(),
        }
    }
}

pub struct AttentionCProjBackwardArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub modules: AttentionBackwardModules<'a>,
    pub saved: BlockForwardSaved<'a>,
    pub projections: AttentionProjectionTensors<'a>,
    pub d_residual_after_attention: &'a DeviceBuffer<f32>,
    pub d_attention_out: &'out mut DeviceBuffer<f32>,
    pub d_attn_c_proj_weight: &'out mut DeviceBuffer<f32>,
    pub scratch: AttentionCProjScratch<'scratch>,
    pub seeds: AttentionBackwardSeeds,
}

pub struct AttentionCoreBackwardArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub module: &'a AttentionModule,
    pub saved: BlockForwardSaved<'a>,
    pub d_attention_out: &'a DeviceBuffer<f32>,
    pub d_qkv: &'out mut DeviceBuffer<f32>,
    pub scratch: AttentionCoreScratch<'scratch>,
}
