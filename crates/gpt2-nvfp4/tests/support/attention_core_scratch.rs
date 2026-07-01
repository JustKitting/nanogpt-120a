#![allow(dead_code)]

use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{AttentionCoreScratch, GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_SEQ_LEN};
use rust_kernels_cuda::attention::CausalAttentionBackwardTcScratch;

pub struct AttentionCoreScratchBuffers {
    softmax_d: DeviceBuffer<f32>,
    q_f32: DeviceBuffer<f32>,
    k_f32: DeviceBuffer<f32>,
    v_f32: DeviceBuffer<f32>,
    g_f32: DeviceBuffer<f32>,
    q: DeviceBuffer<u16>,
    k: DeviceBuffer<u16>,
    v: DeviceBuffer<u16>,
    d_out: DeviceBuffer<u16>,
    scores: DeviceBuffer<f32>,
    dot: DeviceBuffer<f32>,
    p: DeviceBuffer<f32>,
    ds: DeviceBuffer<f32>,
    d_q: DeviceBuffer<f32>,
    d_k: DeviceBuffer<f32>,
    d_v: DeviceBuffer<f32>,
    kda_d_q: DeviceBuffer<f32>,
    kda_d_k: DeviceBuffer<f32>,
    kda_d_v: DeviceBuffer<f32>,
    kda_d_g: DeviceBuffer<f32>,
    kda_d_beta: DeviceBuffer<f32>,
}

impl AttentionCoreScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        let compact = GPT2_BATCH_SIZE * GPT2_SEQ_LEN * GPT2_N_EMBD;
        let square = GPT2_BATCH_SIZE * GPT2_N_HEAD * GPT2_SEQ_LEN * GPT2_SEQ_LEN;
        Ok(Self {
            softmax_d: DeviceBuffer::zeroed(stream, GPT2_BATCH_SIZE * GPT2_N_HEAD * GPT2_SEQ_LEN)?,
            q_f32: DeviceBuffer::zeroed(stream, compact)?,
            k_f32: DeviceBuffer::zeroed(stream, compact)?,
            v_f32: DeviceBuffer::zeroed(stream, compact)?,
            g_f32: DeviceBuffer::zeroed(stream, compact)?,
            q: DeviceBuffer::zeroed(stream, compact)?,
            k: DeviceBuffer::zeroed(stream, compact)?,
            v: DeviceBuffer::zeroed(stream, compact)?,
            d_out: DeviceBuffer::zeroed(stream, compact)?,
            scores: DeviceBuffer::zeroed(stream, square)?,
            dot: DeviceBuffer::zeroed(stream, square)?,
            p: DeviceBuffer::zeroed(stream, square)?,
            ds: DeviceBuffer::zeroed(stream, square)?,
            d_q: DeviceBuffer::zeroed(stream, compact)?,
            d_k: DeviceBuffer::zeroed(stream, compact)?,
            d_v: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_q: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_k: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_v: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_g: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_beta: DeviceBuffer::zeroed(stream, GPT2_BATCH_SIZE * GPT2_N_HEAD * GPT2_SEQ_LEN)?,
        })
    }

    pub fn args(&mut self) -> AttentionCoreScratch<'_> {
        AttentionCoreScratch {
            softmax_d: &mut self.softmax_d,
            tc: CausalAttentionBackwardTcScratch {
                q_f32: &mut self.q_f32,
                k_f32: &mut self.k_f32,
                v_f32: &mut self.v_f32,
                g_f32: &mut self.g_f32,
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
                kda_d_q: &mut self.kda_d_q,
                kda_d_k: &mut self.kda_d_k,
                kda_d_v: &mut self.kda_d_v,
                kda_d_g: &mut self.kda_d_g,
                kda_d_beta: &mut self.kda_d_beta,
            },
        }
    }
}
