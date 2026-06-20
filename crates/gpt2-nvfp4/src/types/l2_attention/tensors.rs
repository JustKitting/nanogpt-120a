use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionTcScratch};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
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
    pub tc_module: &'a F16TcMatmulModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub input_nvfp4: HiddenStateNvfp4<'scratch>,
    pub tc_scratch: CausalAttentionTcScratch<'scratch>,
    pub projections: AttentionProjectionTensors<'a>,
    pub qkv: &'scratch mut DeviceBuffer<f32>,
    pub attention_log_sum_exp: &'scratch mut DeviceBuffer<f32>,
    pub hidden: HiddenStateDevice<'a>,
    pub tape: Option<AttentionForwardTape<'scratch>>,
}
