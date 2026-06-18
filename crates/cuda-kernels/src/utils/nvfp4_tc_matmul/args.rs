use cuda_core::{CudaStream, DeviceBuffer};

use crate::nvfp4::Nvfp4RowwiseDeviceTensor;

pub use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;
pub const NVFP4_TC_MATMUL_K_ALIGNMENT: u32 = 64;

pub struct Nvfp4TcMatmulOperand<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scales: &'a mut DeviceBuffer<f32>,
    pub chunk_amax: &'a mut DeviceBuffer<f32>,
    pub global_scale: f32,
}

pub struct Nvfp4TcMatmulScratch<'a> {
    pub a_padded: &'a mut DeviceBuffer<f32>,
    pub b_t_padded: &'a mut DeviceBuffer<f32>,
    pub a: Nvfp4TcMatmulOperand<'a>,
    pub b_t: Nvfp4TcMatmulOperand<'a>,
}

pub struct Nvfp4TcMatmulArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub quant_module: &'a crate::nvfp4_quant::Nvfp4QuantModule,
    pub a: &'a DeviceBuffer<f32>,
    pub b_t: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub scratch: Nvfp4TcMatmulScratch<'scratch>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub fn nvfp4_tc_matmul_padded_k(k: u32) -> u32 {
    k.div_ceil(NVFP4_TC_MATMUL_K_ALIGNMENT) * NVFP4_TC_MATMUL_K_ALIGNMENT
}

pub fn nvfp4_tc_matmul_elements(rows: u32, k: u32) -> usize {
    rows as usize * nvfp4_tc_matmul_padded_k(k) as usize
}

pub fn nvfp4_tc_matmul_bytes(rows: u32, k: u32) -> usize {
    nvfp4_tc_matmul_elements(rows, k) / 2
}

pub fn nvfp4_tc_matmul_scales(rows: u32, k: u32) -> usize {
    nvfp4_tc_matmul_elements(rows, k) / 16
}

pub fn nvfp4_tc_matmul_chunks(rows: u32, k: u32) -> usize {
    nvfp4_tc_matmul_elements(rows, k) / 32
}

impl<'a> Nvfp4TcMatmulOperand<'a> {
    pub fn reborrow(&mut self) -> Nvfp4TcMatmulOperand<'_> {
        Nvfp4TcMatmulOperand {
            bytes: &mut *self.bytes,
            scales: &mut *self.scales,
            global_scales: &mut *self.global_scales,
            chunk_amax: &mut *self.chunk_amax,
            global_scale: self.global_scale,
        }
    }

    pub(super) fn rowwise(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor {
            bytes: &*self.bytes,
            scales: &*self.scales,
            global_scales: &*self.global_scales,
        }
    }
}

impl<'a> Nvfp4TcMatmulScratch<'a> {
    pub fn reborrow(&mut self) -> Nvfp4TcMatmulScratch<'_> {
        Nvfp4TcMatmulScratch {
            a_padded: &mut *self.a_padded,
            b_t_padded: &mut *self.b_t_padded,
            a: self.a.reborrow(),
            b_t: self.b_t.reborrow(),
        }
    }
}
