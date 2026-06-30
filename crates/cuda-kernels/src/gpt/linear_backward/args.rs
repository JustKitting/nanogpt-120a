use cuda_core::{CudaStream, DeviceBuffer};

use crate::mma::{Nvfp4DeviceScaleMmaWeightTensor, Nvfp4FourSixMmaWeightTensor};
use crate::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use crate::nvfp4_quant::Nvfp4QuantModule;

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

pub struct MsEdenOperandScratch<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scales: &'a mut DeviceBuffer<f32>,
    pub chunk_amax: &'a mut DeviceBuffer<f32>,
    pub global_scale: &'a mut DeviceBuffer<f32>,
}

impl<'a> MsEdenOperandScratch<'a> {
    pub(super) fn rowwise(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor {
            bytes: &*self.bytes,
            scales: &*self.scales,
            global_scales: &*self.global_scales,
        }
    }

    pub(super) fn device_scale_mma_weight(&self) -> Nvfp4DeviceScaleMmaWeightTensor<'_> {
        Nvfp4DeviceScaleMmaWeightTensor {
            bytes: &*self.bytes,
            scales: &*self.scales,
            global_scale: &*self.global_scale,
        }
    }
}

pub struct LinearBackwardMsEdenScratch<'a> {
    pub e_h: MsEdenOperandScratch<'a>,
    pub weight_t_h: MsEdenOperandScratch<'a>,
    pub e_t_h: MsEdenOperandScratch<'a>,
    pub input_t_h: MsEdenOperandScratch<'a>,
}

#[derive(Clone, Copy)]
pub enum LinearBackwardInputTranspose<'a> {
    Fp32(&'a DeviceBuffer<f32>),
    RowwiseNvfp4(Nvfp4RowwiseDeviceTensor<'a>),
}

#[derive(Clone, Copy)]
pub enum LinearBackwardWeightTranspose<'a> {
    Fp32(&'a DeviceBuffer<f32>),
    Nvfp4(Nvfp4DeviceTensor<'a>),
}

pub struct LinearBackwardMsEdenArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub quant_module: &'a Nvfp4QuantModule,
    pub e: &'a DeviceBuffer<f32>,
    pub weight_t: LinearBackwardWeightTranspose<'a>,
    pub input_t: LinearBackwardInputTranspose<'a>,
    pub scratch: LinearBackwardMsEdenScratch<'scratch>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: Option<&'out mut DeviceBuffer<f32>>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
    pub precomputed_e_amax_chunks: Option<u32>,
}
