#![allow(dead_code)]

use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::RowwiseNvfp4Scratch;
use rust_kernels_cuda::attention::CausalAttentionTcScratch;

pub struct RowwiseNvfp4ScratchBuffers {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scales: DeviceBuffer<f32>,
}

impl RowwiseNvfp4ScratchBuffers {
    pub fn new(
        stream: &CudaStream,
        element_count: usize,
        row_count: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, element_count / 2)?,
            scales: DeviceBuffer::zeroed(stream, element_count / 16)?,
            global_scales: DeviceBuffer::zeroed(stream, row_count)?,
        })
    }

    pub fn args(&mut self) -> RowwiseNvfp4Scratch<'_> {
        RowwiseNvfp4Scratch {
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scales: &mut self.global_scales,
        }
    }
}

#[allow(dead_code)]
pub struct CausalAttentionTcScratchBuffers {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    compact_out: DeviceBuffer<f32>,
    chunk_states: DeviceBuffer<u16>,
}

#[allow(dead_code)]
impl CausalAttentionTcScratchBuffers {
    pub fn new(
        stream: &CudaStream,
        compact_len: usize,
        batch_size: usize,
        head_count: usize,
        seq_len: usize,
    ) -> Result<Self, DriverError> {
        let square = batch_size * head_count * seq_len * seq_len;
        Ok(Self {
            q: DeviceBuffer::zeroed(stream, compact_len)?,
            k: DeviceBuffer::zeroed(stream, compact_len)?,
            v: DeviceBuffer::zeroed(stream, compact_len)?,
            scores: DeviceBuffer::zeroed(stream, square)?,
            probs: DeviceBuffer::zeroed(stream, square)?,
            compact_out: DeviceBuffer::zeroed(stream, compact_len)?,
            chunk_states: DeviceBuffer::zeroed(stream, compact_len)?,
        })
    }

    pub fn args(&mut self) -> CausalAttentionTcScratch<'_> {
        CausalAttentionTcScratch {
            q: &mut self.q,
            k: &mut self.k,
            v: &mut self.v,
            scores: &mut self.scores,
            probs: &mut self.probs,
            compact_out: &mut self.compact_out,
            chunk_states: &mut self.chunk_states,
        }
    }
}
