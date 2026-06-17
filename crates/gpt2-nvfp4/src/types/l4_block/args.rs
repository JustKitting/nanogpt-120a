use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use crate::types::{
    AttentionProjectionTensors, BlockForwardTape, HiddenStateDevice, HiddenStateNvfp4,
    LayerNormTensors, MlpActivationNvfp4, MlpDownTensors, MlpUpTensors,
};

pub struct BlockForwardArgs<'a, 'scratch> {
    pub attention_module: &'a AttentionModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub layer_norm_module: &'a LayerNormModule,
    pub mlp_module: &'a MlpModule,
    pub hidden_nvfp4: HiddenStateNvfp4<'scratch>,
    pub mlp_activation_nvfp4: MlpActivationNvfp4<'scratch>,
    pub projections: AttentionProjectionTensors<'a>,
    pub ln_1: LayerNormTensors<'a>,
    pub ln_2: LayerNormTensors<'a>,
    pub mlp_up: MlpUpTensors<'a>,
    pub mlp_down: MlpDownTensors<'a>,
    pub qkv: &'scratch mut DeviceBuffer<f32>,
    pub attention_log_sum_exp: &'scratch mut DeviceBuffer<f32>,
    pub mlp_pre_activation: &'scratch mut DeviceBuffer<f32>,
    pub mlp_activation: &'scratch mut DeviceBuffer<f32>,
    pub hidden: HiddenStateDevice<'a>,
    pub tape: Option<BlockForwardTape<'scratch>>,
}
