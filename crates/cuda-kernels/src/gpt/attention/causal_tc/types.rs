use cuda_core::{CudaStream, DeviceBuffer};

use crate::f16_tc_matmul::F16TcMatmulModule;

pub struct CausalAttentionTcScratch<'a> {
    pub q: &'a mut DeviceBuffer<f32>,
    pub k: &'a mut DeviceBuffer<f32>,
    pub v: &'a mut DeviceBuffer<f32>,
    pub scores: &'a mut DeviceBuffer<f32>,
    pub probs: &'a mut DeviceBuffer<f32>,
    pub compact_out: &'a mut DeviceBuffer<f32>,
}

pub struct CausalAttentionTcArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub tc_module: &'a F16TcMatmulModule,
    pub qkv: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub attention_out_f16: Option<&'out mut DeviceBuffer<u16>>,
    pub log_sum_exp: &'out mut DeviceBuffer<f32>,
    pub scratch: CausalAttentionTcScratch<'scratch>,
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
}

impl<'a> CausalAttentionTcScratch<'a> {
    pub fn reborrow(&mut self) -> CausalAttentionTcScratch<'_> {
        CausalAttentionTcScratch {
            q: self.q,
            k: self.k,
            v: self.v,
            scores: self.scores,
            probs: self.probs,
            compact_out: self.compact_out,
        }
    }
}
