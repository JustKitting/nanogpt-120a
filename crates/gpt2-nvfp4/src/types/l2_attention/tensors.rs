use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use super::tape::AttentionForwardTape;
use crate::types::{HiddenStateDevice, HiddenStateNvfp4};

#[derive(Clone, Copy)]
pub struct AttentionProjectionTensors<'a> {
    pub qkv_weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub qkv_bias: Nvfp4DeviceTensor<'a>,
    pub c_proj_weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub c_proj_bias: Nvfp4DeviceTensor<'a>,
}

pub struct AttentionForwardArgs<'a, 'scratch> {
    pub module: &'a AttentionModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub input_nvfp4: HiddenStateNvfp4<'scratch>,
    pub projections: AttentionProjectionTensors<'a>,
    pub qkv: &'scratch mut DeviceBuffer<f32>,
    pub attention_log_sum_exp: &'scratch mut DeviceBuffer<f32>,
    pub hidden: HiddenStateDevice<'a>,
    pub tape: Option<AttentionForwardTape<'scratch>>,
}
