use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    BlockForwardSaved, GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS, HiddenState, LayerNormSaved,
    QkvActivation,
};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use crate::data::{self, E2M1_MIN_PAIR, E4M3_ONE};

pub struct SavedBuffers {
    hidden_bytes: DeviceBuffer<u8>,
    hidden_scales: DeviceBuffer<u8>,
    hidden_globals: DeviceBuffer<f32>,
    hidden_f16: DeviceBuffer<u16>,
    qkv: DeviceBuffer<u16>,
    log_sum_exp: DeviceBuffer<f32>,
    mean: DeviceBuffer<f32>,
    inv_std: DeviceBuffer<f32>,
}

impl SavedBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            hidden_bytes: DeviceBuffer::from_host(
                stream,
                &vec![E2M1_MIN_PAIR; HiddenState::LEN / 2],
            )?,
            hidden_scales: DeviceBuffer::from_host(stream, &vec![E4M3_ONE; HiddenState::LEN / 16])?,
            hidden_globals: DeviceBuffer::from_host(stream, &data::row_global_scales())?,
            hidden_f16: DeviceBuffer::from_host(stream, &vec![0x2e66_u16; HiddenState::LEN])?,
            qkv: DeviceBuffer::from_host(stream, &vec![0x3c00_u16; QkvActivation::LEN])?,
            log_sum_exp: DeviceBuffer::from_host(stream, &data::attention_log_sum_exp_values())?,
            mean: DeviceBuffer::zeroed(stream, GPT2_TOKEN_ROWS)?,
            inv_std: DeviceBuffer::from_host(stream, &data::inv_std_values())?,
        })
    }

    pub fn block(&self) -> BlockForwardSaved<'_> {
        let rowwise = Nvfp4RowwiseDeviceTensor::new(
            &self.hidden_bytes,
            &self.hidden_scales,
            &self.hidden_globals,
        );
        let ln = LayerNormSaved {
            row_count: GPT2_TOKEN_ROWS as u32,
            residual: &self.hidden_f16,
            mean: &self.mean,
            inv_std: &self.inv_std,
        };
        BlockForwardSaved {
            batch_size: GPT2_BATCH_SIZE as u32,
            seq_len: GPT2_SEQ_LEN as u32,
            row_count: GPT2_TOKEN_ROWS as u32,
            ln_1: ln,
            qkv_input_nvfp4: rowwise,
            qkv: &self.qkv,
            attention_out: &self.hidden_f16,
            attention_log_sum_exp: &self.log_sum_exp,
            c_proj_input_nvfp4: rowwise,
            ln_2: ln,
            mlp_up_input_nvfp4: rowwise,
            mlp_up: &self.hidden_f16,
            mlp_down_input_nvfp4: rowwise,
        }
    }
}
