use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use crate::random::InitRng;
use crate::{GPT2_N_LAYER, Gpt2Config};

use super::{
    AttentionProjectionTensors, BlockForwardArgs, EmbeddingWeights, Gpt2BlockWeights,
    HiddenStateDevice, HiddenStateNvfp4, LayerNormTensors, LayerNormWeights, MlpActivationNvfp4,
    MlpDownTensors, MlpUpTensors, TokenEmbeddingArgs,
};

pub struct Gpt2ForwardArgs<'a> {
    pub embeddings: TokenEmbeddingArgs<'a>,
    pub attention_module: &'a AttentionModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub layer_norm_module: &'a LayerNormModule,
    pub mlp_module: &'a MlpModule,
    pub hidden_nvfp4: HiddenStateNvfp4<'a>,
    pub mlp_activation_nvfp4: MlpActivationNvfp4<'a>,
    pub attention_qkv_weights: [Nvfp4FourSixMmaWeightTensor<'a>; GPT2_N_LAYER],
    pub attention_qkv_biases: [Nvfp4DeviceTensor<'a>; GPT2_N_LAYER],
    pub attention_c_proj_weights: [Nvfp4FourSixMmaWeightTensor<'a>; GPT2_N_LAYER],
    pub attention_c_proj_biases: [Nvfp4DeviceTensor<'a>; GPT2_N_LAYER],
    pub block_ln_1: [LayerNormTensors<'a>; GPT2_N_LAYER],
    pub block_ln_2: [LayerNormTensors<'a>; GPT2_N_LAYER],
    pub mlp_up: [MlpUpTensors<'a>; GPT2_N_LAYER],
    pub mlp_down: [MlpDownTensors<'a>; GPT2_N_LAYER],
    pub ln_f: LayerNormTensors<'a>,
    pub attention_qkv: &'a mut DeviceBuffer<f32>,
    pub mlp_activation: &'a mut DeviceBuffer<f32>,
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
            quant_module,
            layer_norm_module,
            mlp_module,
            mut hidden_nvfp4,
            mut mlp_activation_nvfp4,
            attention_qkv_weights,
            attention_qkv_biases,
            attention_c_proj_weights,
            attention_c_proj_biases,
            block_ln_1,
            block_ln_2,
            mlp_up,
            mlp_down,
            ln_f,
            attention_qkv,
            mlp_activation,
        } = args;

        let mut hidden = self.embeddings.forward(embeddings)?;

        for (block_index, block) in self.h.iter().enumerate() {
            hidden = block.forward(BlockForwardArgs {
                attention_module,
                quant_module,
                layer_norm_module,
                mlp_module,
                hidden_nvfp4: hidden_nvfp4.reborrow(),
                mlp_activation_nvfp4: mlp_activation_nvfp4.reborrow(),
                projections: AttentionProjectionTensors {
                    qkv_weight: attention_qkv_weights[block_index],
                    qkv_bias: attention_qkv_biases[block_index],
                    c_proj_weight: attention_c_proj_weights[block_index],
                    c_proj_bias: attention_c_proj_biases[block_index],
                },
                ln_1: block_ln_1[block_index],
                ln_2: block_ln_2[block_index],
                mlp_up: mlp_up[block_index],
                mlp_down: mlp_down[block_index],
                qkv: &mut *attention_qkv,
                mlp_activation: &mut *mlp_activation,
                hidden,
            })?;
        }

        self.ln_f.forward(LayerNormWeights::input_from_block(
            layer_norm_module,
            ln_f,
            hidden,
        ))
    }
}
