use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};

use crate::f16_tc_matmul::{F16TcMatmulModule, F16TcMatmulScratch};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CausalAttentionBackwardTcParams {
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
    pub scale: f32,
}

unsafe impl DeviceCopy for CausalAttentionBackwardTcParams {}

pub struct CausalAttentionBackwardTcScratch<'a> {
    pub q: &'a mut DeviceBuffer<f32>,
    pub k: &'a mut DeviceBuffer<f32>,
    pub v: &'a mut DeviceBuffer<f32>,
    pub d_out: &'a mut DeviceBuffer<f32>,
    pub q_t: &'a mut DeviceBuffer<f32>,
    pub k_t: &'a mut DeviceBuffer<f32>,
    pub d_out_t: &'a mut DeviceBuffer<f32>,
    pub p_t: &'a mut DeviceBuffer<f32>,
    pub ds_t: &'a mut DeviceBuffer<f32>,
    pub scores: &'a mut DeviceBuffer<f32>,
    pub dot: &'a mut DeviceBuffer<f32>,
    pub p: &'a mut DeviceBuffer<f32>,
    pub ds: &'a mut DeviceBuffer<f32>,
    pub d_q: &'a mut DeviceBuffer<f32>,
    pub d_k: &'a mut DeviceBuffer<f32>,
    pub d_v: &'a mut DeviceBuffer<f32>,
    pub matmul: F16TcMatmulScratch<'a>,
}

pub struct CausalAttentionBackwardTcArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub tc_module: &'a F16TcMatmulModule,
    pub qkv: &'a DeviceBuffer<f32>,
    pub attention_out: &'a DeviceBuffer<f32>,
    pub d_out: &'a DeviceBuffer<f32>,
    pub log_sum_exp: &'a DeviceBuffer<f32>,
    pub softmax_d: &'scratch mut DeviceBuffer<f32>,
    pub d_qkv: &'out mut DeviceBuffer<f32>,
    pub scratch: CausalAttentionBackwardTcScratch<'scratch>,
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
}

impl<'a> CausalAttentionBackwardTcScratch<'a> {
    pub fn reborrow(&mut self) -> CausalAttentionBackwardTcScratch<'_> {
        CausalAttentionBackwardTcScratch {
            q: self.q,
            k: self.k,
            v: self.v,
            d_out: self.d_out,
            q_t: self.q_t,
            k_t: self.k_t,
            d_out_t: self.d_out_t,
            p_t: self.p_t,
            ds_t: self.ds_t,
            scores: self.scores,
            dot: self.dot,
            p: self.p,
            ds: self.ds,
            d_q: self.d_q,
            d_k: self.d_k,
            d_v: self.d_v,
            matmul: self.matmul.reborrow(),
        }
    }
}
