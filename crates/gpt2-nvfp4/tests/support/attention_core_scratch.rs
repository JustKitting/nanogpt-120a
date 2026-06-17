use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{AttentionCoreScratch, GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_SEQ_LEN};
use rust_kernels_cuda::attention::CausalAttentionBackwardTcScratch;
use rust_kernels_cuda::f16_tc_matmul::{F16TcMatmulScratch, f16_tc_matmul_elements};

pub struct AttentionCoreScratchBuffers {
    softmax_d: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    q_t: DeviceBuffer<f32>,
    k_t: DeviceBuffer<f32>,
    d_out_t: DeviceBuffer<f32>,
    p_t: DeviceBuffer<f32>,
    ds_t: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    dot: DeviceBuffer<f32>,
    p: DeviceBuffer<f32>,
    ds: DeviceBuffer<f32>,
    d_q: DeviceBuffer<f32>,
    d_k: DeviceBuffer<f32>,
    d_v: DeviceBuffer<f32>,
    a_padded: DeviceBuffer<f32>,
    b_t_padded: DeviceBuffer<f32>,
    a_halves: DeviceBuffer<u16>,
    b_t_halves: DeviceBuffer<u16>,
}

impl AttentionCoreScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        let compact = GPT2_BATCH_SIZE * GPT2_SEQ_LEN * GPT2_N_EMBD;
        let square = GPT2_BATCH_SIZE * GPT2_N_HEAD * GPT2_SEQ_LEN * GPT2_SEQ_LEN;
        let matmul = f16_tc_matmul_elements(matmul_rows() as u32, matmul_k() as u32);
        Ok(Self {
            softmax_d: zero(stream, GPT2_BATCH_SIZE * GPT2_N_HEAD * GPT2_SEQ_LEN)?,
            q: zero(stream, compact)?,
            k: zero(stream, compact)?,
            v: zero(stream, compact)?,
            d_out: zero(stream, compact)?,
            q_t: zero(stream, compact)?,
            k_t: zero(stream, compact)?,
            d_out_t: zero(stream, compact)?,
            p_t: zero(stream, square)?,
            ds_t: zero(stream, square)?,
            scores: zero(stream, square)?,
            dot: zero(stream, square)?,
            p: zero(stream, square)?,
            ds: zero(stream, square)?,
            d_q: zero(stream, compact)?,
            d_k: zero(stream, compact)?,
            d_v: zero(stream, compact)?,
            a_padded: zero(stream, matmul)?,
            b_t_padded: zero(stream, matmul)?,
            a_halves: zero(stream, matmul)?,
            b_t_halves: zero(stream, matmul)?,
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
                q_t: &mut self.q_t,
                k_t: &mut self.k_t,
                d_out_t: &mut self.d_out_t,
                p_t: &mut self.p_t,
                ds_t: &mut self.ds_t,
                scores: &mut self.scores,
                dot: &mut self.dot,
                p: &mut self.p,
                ds: &mut self.ds,
                d_q: &mut self.d_q,
                d_k: &mut self.d_k,
                d_v: &mut self.d_v,
                matmul: F16TcMatmulScratch {
                    a_padded: &mut self.a_padded,
                    b_t_padded: &mut self.b_t_padded,
                    a_halves: &mut self.a_halves,
                    b_t_halves: &mut self.b_t_halves,
                },
            },
        }
    }
}

fn zero<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    len: usize,
) -> Result<DeviceBuffer<T>, DriverError> {
    DeviceBuffer::zeroed(stream, len)
}

fn matmul_rows() -> usize {
    GPT2_BATCH_SIZE * GPT2_N_HEAD * GPT2_SEQ_LEN.max(GPT2_N_EMBD / GPT2_N_HEAD)
}

fn matmul_k() -> usize {
    GPT2_SEQ_LEN.max(GPT2_N_EMBD / GPT2_N_HEAD)
}
