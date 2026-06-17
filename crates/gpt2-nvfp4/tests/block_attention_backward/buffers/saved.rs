use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{BlockForwardSaved, GPT2_CONTEXT_LEN, HiddenState, LayerNormSaved, QkvActivation};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use crate::data::{self, E2M1_MIN_PAIR, E4M3_ONE};

pub struct SavedBuffers {
    hidden_bytes: DeviceBuffer<u8>,
    hidden_scales: DeviceBuffer<u8>,
    hidden_globals: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    lse: DeviceBuffer<f32>,
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
            hidden: DeviceBuffer::from_host(stream, &data::hidden_values())?,
            qkv: DeviceBuffer::from_host(stream, &vec![0.0; QkvActivation::LEN])?,
            lse: DeviceBuffer::from_host(stream, &data::attention_lse_values())?,
            mean: DeviceBuffer::zeroed(stream, GPT2_CONTEXT_LEN)?,
            inv_std: DeviceBuffer::from_host(stream, &data::inv_std_values())?,
        })
    }

    pub fn block(&self) -> BlockForwardSaved<'_> {
        let rowwise = Nvfp4RowwiseDeviceTensor {
            bytes: &self.hidden_bytes,
            scales: &self.hidden_scales,
            global_scales: &self.hidden_globals,
        };
        let ln = LayerNormSaved {
            residual: &self.hidden,
            normalized: &self.hidden,
            mean: &self.mean,
            inv_std: &self.inv_std,
        };
        BlockForwardSaved {
            residual_in: &self.hidden,
            ln_1: ln,
            qkv_input_nvfp4: rowwise,
            qkv: &self.qkv,
            attention_out: &self.hidden,
            attention_lse: &self.lse,
            c_proj_input_nvfp4: rowwise,
            residual_after_attention: &self.hidden,
            ln_2: ln,
            mlp_up_input_nvfp4: rowwise,
            mlp_up: &self.hidden,
            mlp_relu2: &self.hidden,
            mlp_down_input_nvfp4: rowwise,
            residual_out: &self.hidden,
        }
    }
}
