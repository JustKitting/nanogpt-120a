use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::lm_head::LmHeadModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use crate::GPT2_N_LAYER;
use crate::types::{
    Gpt2ForwardTape, HiddenStateNvfp4, LayerNormTensors, MlpActivationNvfp4, MlpDownTensors,
    MlpUpTensors, TokenEmbeddingArgs,
};

pub struct Gpt2ForwardArgs<'a> {
    pub embeddings: TokenEmbeddingArgs<'a>,
    pub attention_module: &'a AttentionModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub layer_norm_module: &'a LayerNormModule,
    pub mlp_module: &'a MlpModule,
    pub lm_head_module: &'a LmHeadModule,
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
    pub logits: &'a mut DeviceBuffer<f32>,
    pub tape: Option<Gpt2ForwardTape<'a>>,
}
