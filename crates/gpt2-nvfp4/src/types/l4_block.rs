use crate::random::InitRng;
use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use super::{
    AttentionInputNvfp4, AttentionProjectionTensors, AttentionWeights, HiddenStateDevice,
    LayerNormWeights, MlpWeights,
};

pub struct BlockForwardArgs<'a, 'scratch> {
    pub attention_module: &'a AttentionModule,
    pub attention_quant_module: &'a Nvfp4QuantModule,
    pub attention_input_nvfp4: AttentionInputNvfp4<'scratch>,
    pub projections: AttentionProjectionTensors<'a>,
    pub qkv: &'scratch mut DeviceBuffer<f32>,
    pub hidden: HiddenStateDevice<'a>,
}

#[derive(Clone, Debug)]
pub struct Gpt2BlockWeights {
    pub ln_1: LayerNormWeights,
    pub attn: AttentionWeights,
    pub ln_2: LayerNormWeights,
    pub mlp: MlpWeights,
}

impl Gpt2BlockWeights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        Self {
            ln_1: LayerNormWeights::init(),
            attn: AttentionWeights::init(rng),
            ln_2: LayerNormWeights::init(),
            mlp: MlpWeights::init(rng),
        }
    }

    pub fn forward<'a, 'scratch>(
        &self,
        args: BlockForwardArgs<'a, 'scratch>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let hidden = AttentionWeights::forward(AttentionWeights::input_from_embeddings(
            args.attention_module,
            args.attention_quant_module,
            args.attention_input_nvfp4,
            args.projections,
            args.qkv,
            args.hidden,
        ))?;
        self.mlp.forward(MlpWeights::input_from_attention(hidden))
    }
}
