use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::mma::Nvfp4DeviceScaleMmaWeightTensor;
use crate::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use crate::nvfp4_quant::Nvfp4QuantModule;

pub struct MsEdenOperandScratch<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scales: &'a mut DeviceBuffer<f32>,
    pub chunk_amax: &'a mut DeviceBuffer<f32>,
    pub global_scale: &'a mut DeviceBuffer<f32>,
}

impl<'a> MsEdenOperandScratch<'a> {
    pub(crate) fn rowwise(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor {
            bytes: &*self.bytes,
            scales: &*self.scales,
            global_scales: &*self.global_scales,
        }
    }

    pub(crate) fn device_scale_mma_weight(&self) -> Nvfp4DeviceScaleMmaWeightTensor<'_> {
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

pub struct LinearBackwardMsEdenScratchBuffers {
    pub e_h: MsEdenOperandScratchBuffer,
    pub weight_t_h: MsEdenOperandScratchBuffer,
    pub e_t_h: MsEdenOperandScratchBuffer,
    pub input_t_h: MsEdenOperandScratchBuffer,
}

impl LinearBackwardMsEdenScratchBuffers {
    pub fn new(
        stream: &CudaStream,
        token_count: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            e_h: MsEdenOperandScratchBuffer::new(stream, token_count * output_dim, token_count)?,
            weight_t_h: MsEdenOperandScratchBuffer::new(stream, input_dim * output_dim, input_dim)?,
            e_t_h: MsEdenOperandScratchBuffer::new(stream, output_dim * token_count, output_dim)?,
            input_t_h: MsEdenOperandScratchBuffer::new(stream, input_dim * token_count, input_dim)?,
        })
    }

    pub fn as_args(&mut self) -> LinearBackwardMsEdenScratch<'_> {
        LinearBackwardMsEdenScratch {
            e_h: self.e_h.as_arg(),
            weight_t_h: self.weight_t_h.as_arg(),
            e_t_h: self.e_t_h.as_arg(),
            input_t_h: self.input_t_h.as_arg(),
        }
    }
}

pub struct MsEdenOperandScratchBuffer {
    pub bytes: DeviceBuffer<u8>,
    pub scales: DeviceBuffer<u8>,
    pub global_scales: DeviceBuffer<f32>,
    pub chunk_amax: DeviceBuffer<f32>,
    pub global_scale: DeviceBuffer<f32>,
}

impl MsEdenOperandScratchBuffer {
    pub fn new(stream: &CudaStream, elements: usize, rows: usize) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, elements / 2)?,
            scales: DeviceBuffer::zeroed(stream, elements / 16)?,
            global_scales: DeviceBuffer::zeroed(stream, rows)?,
            chunk_amax: DeviceBuffer::zeroed(stream, elements / 32)?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    pub fn as_arg(&mut self) -> MsEdenOperandScratch<'_> {
        MsEdenOperandScratch {
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scales: &mut self.global_scales,
            chunk_amax: &mut self.chunk_amax,
            global_scale: &mut self.global_scale,
        }
    }
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
