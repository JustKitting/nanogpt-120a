use crate::random::InitRng;
use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use super::{
    AttentionProjectionTensors, AttentionWeights, HiddenStateDevice, HiddenStateNvfp4,
    LayerNormTensors, LayerNormWeights, MlpUpTensors, MlpWeights,
};

pub struct BlockForwardArgs<'a, 'scratch> {
    pub attention_module: &'a AttentionModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub layer_norm_module: &'a LayerNormModule,
    pub mlp_module: &'a MlpModule,
    pub hidden_nvfp4: HiddenStateNvfp4<'scratch>,
    pub projections: AttentionProjectionTensors<'a>,
    pub ln_2: LayerNormTensors<'a>,
    pub mlp_up: MlpUpTensors<'a>,
    pub qkv: &'scratch mut DeviceBuffer<f32>,
    pub mlp_activation: &'scratch mut DeviceBuffer<f32>,
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
        let mut hidden_nvfp4 = args.hidden_nvfp4;
        let hidden = AttentionWeights::forward(AttentionWeights::input_from_embeddings(
            args.attention_module,
            args.quant_module,
            hidden_nvfp4.reborrow(),
            args.projections,
            args.qkv,
            args.hidden,
        ))?;

        let hidden = self.ln_2.forward(LayerNormWeights::input_from_block(
            args.layer_norm_module,
            args.ln_2,
            hidden,
        ))?;

        MlpWeights::forward(MlpWeights::input_from_attention(
            args.mlp_module,
            args.quant_module,
            hidden_nvfp4.reborrow(),
            args.mlp_up,
            args.mlp_activation,
            hidden,
        ))
    }
}
