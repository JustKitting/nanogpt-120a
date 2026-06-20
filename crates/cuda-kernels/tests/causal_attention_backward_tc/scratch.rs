use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::attention::CausalAttentionBackwardTcScratch;

use super::shape::{HEAD_DIM, HEADS, TOKEN_COUNT};

pub struct TcScratchBuffers {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    dot: DeviceBuffer<f32>,
    p: DeviceBuffer<f32>,
    ds: DeviceBuffer<f32>,
    d_q: DeviceBuffer<f32>,
    d_k: DeviceBuffer<f32>,
    d_v: DeviceBuffer<f32>,
}

impl TcScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        let compact = HEADS * TOKEN_COUNT * HEAD_DIM;
        let square = HEADS * TOKEN_COUNT * TOKEN_COUNT;
        Ok(Self {
            q: zero(stream, compact)?,
            k: zero(stream, compact)?,
            v: zero(stream, compact)?,
            d_out: zero(stream, compact)?,
            scores: zero(stream, square)?,
            dot: zero(stream, square)?,
            p: zero(stream, square)?,
            ds: zero(stream, square)?,
            d_q: zero(stream, compact)?,
            d_k: zero(stream, compact)?,
            d_v: zero(stream, compact)?,
        })
    }

    pub fn args(&mut self) -> CausalAttentionBackwardTcScratch<'_> {
        CausalAttentionBackwardTcScratch {
            q: &mut self.q,
            k: &mut self.k,
            v: &mut self.v,
            d_out: &mut self.d_out,
            scores: &mut self.scores,
            dot: &mut self.dot,
            p: &mut self.p,
            ds: &mut self.ds,
            d_q: &mut self.d_q,
            d_k: &mut self.d_k,
            d_v: &mut self.d_v,
        }
    }
}

fn zero<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    len: usize,
) -> Result<DeviceBuffer<T>, DriverError> {
    DeviceBuffer::zeroed(stream, len)
}
