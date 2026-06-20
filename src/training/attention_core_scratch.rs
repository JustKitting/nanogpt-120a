use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{AttentionCoreScratch, GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_SEQ_LEN};
use rust_kernels_cuda::attention::{CausalAttentionBackwardTcScratch, CausalAttentionTcScratch};

pub struct AttentionCoreScratchBuffers {
    softmax_d: DeviceBuffer<f32>,
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

impl AttentionCoreScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        let compact = GPT2_BATCH_SIZE * GPT2_SEQ_LEN * GPT2_N_EMBD;
        let square = GPT2_BATCH_SIZE * GPT2_N_HEAD * GPT2_SEQ_LEN * GPT2_SEQ_LEN;
        Ok(Self {
            softmax_d: zero(stream, GPT2_BATCH_SIZE * GPT2_N_HEAD * GPT2_SEQ_LEN)?,
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

    pub fn args(&mut self) -> AttentionCoreScratch<'_> {
        AttentionCoreScratch {
            softmax_d: &mut self.softmax_d,
            tc: CausalAttentionBackwardTcScratch {
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
            },
        }
    }

    pub fn forward_tc(&mut self) -> CausalAttentionTcScratch<'_> {
        CausalAttentionTcScratch {
            q: &mut self.q,
            k: &mut self.k,
            v: &mut self.v,
            scores: &mut self.scores,
            probs: &mut self.p,
            compact_out: &mut self.d_out,
        }
    }
}

fn zero<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    len: usize,
) -> Result<DeviceBuffer<T>, DriverError> {
    DeviceBuffer::zeroed(stream, len)
}
