#[path = "args/ms_eden.rs"]
mod ms_eden;

use cuda_core::{CudaStream, DeviceBuffer};

use crate::mma::{Nvfp4DeviceScaleMmaWeightTensor, Nvfp4FourSixMmaWeightTensor};
use crate::nvfp4::Nvfp4RowwiseDeviceTensor;

pub use ms_eden::{
    LinearBackwardInputTranspose, LinearBackwardMsEdenArgs, LinearBackwardMsEdenScratch,
    LinearBackwardWeightTranspose, MsEdenOperandScratch, MsEdenOperandScratchBuffer,
};

pub struct LinearBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub e_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight_t_h: Nvfp4FourSixMmaWeightTensor<'a>,
    pub e_t_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub input_t_h: Nvfp4FourSixMmaWeightTensor<'a>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: Option<&'out mut DeviceBuffer<f32>>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
}

pub struct LinearBackwardDeviceScaleArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub e_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight_t_h: Nvfp4DeviceScaleMmaWeightTensor<'a>,
    pub e_t_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub input_t_h: Nvfp4DeviceScaleMmaWeightTensor<'a>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
}
