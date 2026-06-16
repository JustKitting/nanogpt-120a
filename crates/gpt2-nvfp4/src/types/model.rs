use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use crate::random::InitRng;
use crate::{GPT2_N_LAYER, Gpt2Config};

use super::{
    AttentionInputNvfp4, AttentionProjectionTensors, BlockForwardArgs, EmbeddingWeights,
    Gpt2BlockWeights, HiddenStateDevice, LayerNormWeights, TokenEmbeddingArgs,
};

pub struct Gpt2ForwardArgs<'a> {
    pub embeddings: TokenEmbeddingArgs<'a>,
    pub attention_module: &'a AttentionModule,
    pub attention_quant_module: &'a Nvfp4QuantModule,
    pub attention_input_nvfp4: AttentionInputNvfp4<'a>,
    pub attention_qkv_weights: [Nvfp4FourSixMmaWeightTensor<'a>; GPT2_N_LAYER],
    pub attention_qkv_biases: [Nvfp4DeviceTensor<'a>; GPT2_N_LAYER],
    pub attention_c_proj_weights: [Nvfp4FourSixMmaWeightTensor<'a>; GPT2_N_LAYER],
    pub attention_c_proj_biases: [Nvfp4DeviceTensor<'a>; GPT2_N_LAYER],
    pub attention_qkv: &'a mut DeviceBuffer<f32>,
}

#[derive(Clone, Debug)]
pub struct Gpt2 {
    weights: Option<Gpt2Weights>,
}

impl Gpt2 {
    pub const fn new() -> Self {
        Self { weights: None }
    }

    pub fn init(&mut self, seed: u64) {
        let mut rng = InitRng::new(seed);
        self.weights = Some(Gpt2Weights::init(&mut rng));
    }

    pub fn weights(&self) -> Option<&Gpt2Weights> {
        self.weights.as_ref()
    }

    pub fn weights_mut(&mut self) -> Option<&mut Gpt2Weights> {
        self.weights.as_mut()
    }

    pub fn forward_embeddings<'a>(
        &self,
        args: TokenEmbeddingArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        self.weights()
            .expect("Gpt2::init must be called before forward_embeddings")
            .forward_embeddings(args)
    }

    pub fn forward<'a>(
        &self,
        args: Gpt2ForwardArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        self.weights()
            .expect("Gpt2::init must be called before forward")
            .forward(args)
    }
}

impl Default for Gpt2 {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct Gpt2Weights {
    pub config: Gpt2Config,
    pub embeddings: EmbeddingWeights,
    pub h: [Gpt2BlockWeights; GPT2_N_LAYER],
    pub ln_f: LayerNormWeights,
}

impl Gpt2Weights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        Self {
            config: Gpt2Config::gpt2_124m(),
            embeddings: EmbeddingWeights::init(rng),
            h: std::array::from_fn(|_| Gpt2BlockWeights::init(rng)),
            ln_f: LayerNormWeights::init(),
        }
    }

    pub fn forward_embeddings<'a>(
        &self,
        args: TokenEmbeddingArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        self.embeddings.forward(args)
    }

    pub fn forward<'a>(
        &self,
        args: Gpt2ForwardArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let Gpt2ForwardArgs {
            embeddings,
            attention_module,
            attention_quant_module,
            mut attention_input_nvfp4,
            attention_qkv_weights,
            attention_qkv_biases,
            attention_c_proj_weights,
            attention_c_proj_biases,
            attention_qkv,
        } = args;

        let mut hidden = self.embeddings.forward(embeddings)?;

        for (block_index, block) in self.h.iter().enumerate() {
            hidden = block.forward(BlockForwardArgs {
                attention_module,
                attention_quant_module,
                attention_input_nvfp4: attention_input_nvfp4.reborrow(),
                projections: AttentionProjectionTensors {
                    qkv_weight: attention_qkv_weights[block_index],
                    qkv_bias: attention_qkv_biases[block_index],
                    c_proj_weight: attention_c_proj_weights[block_index],
                    c_proj_bias: attention_c_proj_biases[block_index],
                },
                qkv: &mut *attention_qkv,
                hidden,
            })?;
        }

        self.ln_f
            .forward(LayerNormWeights::input_from_block(hidden))
    }
}
