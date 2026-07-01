use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{BlockForwardSaved, GPT2_TOKEN_ROWS, HiddenState, QkvActivation};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use crate::data::{self, E2M1_MIN_PAIR, E4M3_ONE};
use crate::nvfp4_common::filled_u8;
use crate::saved_block::{SavedBlockParts, saved_block};

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
            hidden_bytes: filled_u8(stream, HiddenState::LEN / 2, E2M1_MIN_PAIR)?,
            hidden_scales: filled_u8(stream, HiddenState::LEN / 16, E4M3_ONE)?,
            hidden_globals: DeviceBuffer::from_host(stream, &data::row_global_scales())?,
            hidden_f16: DeviceBuffer::from_host(stream, &vec![0x2e66_u16; HiddenState::LEN])?,
            qkv: DeviceBuffer::from_host(stream, &vec![0x3c00_u16; QkvActivation::LEN])?,
            log_sum_exp: DeviceBuffer::from_host(stream, &crate::common::attention_log_sum_exp_values())?,
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
        saved_block(SavedBlockParts {
            rowwise,
            residual: &self.hidden_f16,
            mean: &self.mean,
            inv_std: &self.inv_std,
            qkv: &self.qkv,
            attention_out: &self.hidden_f16,
            attention_log_sum_exp: &self.log_sum_exp,
            mlp_up: &self.hidden_f16,
        })
    }
}
