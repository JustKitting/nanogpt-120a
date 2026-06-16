use crate::random::InitRng;
use cuda_core::DriverError;
use rust_kernels_cuda::attention::AttentionModule;

use super::{AttentionWeights, HiddenStateDevice, LayerNormWeights, MlpWeights};

pub struct BlockForwardArgs<'a> {
    pub attention_module: &'a AttentionModule,
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

    pub fn forward<'a>(
        &self,
        args: BlockForwardArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let hidden = self.attn.forward(AttentionWeights::input_from_embeddings(
            args.attention_module,
            args.hidden,
        ))?;
        self.mlp.forward(MlpWeights::input_from_attention(hidden))
    }
}
