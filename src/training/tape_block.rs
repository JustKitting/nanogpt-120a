use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionLogSumExp, BlockForwardSaved, BlockForwardTape, GPT2_TOKEN_ROWS, HiddenState,
    MlpActivation, QkvActivation,
};

use super::tape_leaf::{LayerNormTapeBuffers, RowwiseTapeBuffers, zero};

pub struct BlockTapeBuffers {
    ln_1: LayerNormTapeBuffers,
    qkv_input: RowwiseTapeBuffers,
    qkv: DeviceBuffer<u16>,
    attention_out: DeviceBuffer<u16>,
    attention_log_sum_exp: DeviceBuffer<f32>,
    c_proj_input: RowwiseTapeBuffers,
    ln_2: LayerNormTapeBuffers,
    mlp_up_input: RowwiseTapeBuffers,
    mlp_up: DeviceBuffer<u16>,
    mlp_down_input: RowwiseTapeBuffers,
}

impl BlockTapeBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            ln_1: LayerNormTapeBuffers::new(stream)?,
            qkv_input: RowwiseTapeBuffers::new(stream, HiddenState::LEN, GPT2_TOKEN_ROWS)?,
            qkv: DeviceBuffer::zeroed(stream, QkvActivation::LEN)?,
            attention_out: DeviceBuffer::zeroed(stream, HiddenState::LEN)?,
            attention_log_sum_exp: zero(stream, AttentionLogSumExp::LEN)?,
            c_proj_input: RowwiseTapeBuffers::new(stream, HiddenState::LEN, GPT2_TOKEN_ROWS)?,
            ln_2: LayerNormTapeBuffers::new(stream)?,
            mlp_up_input: RowwiseTapeBuffers::new(stream, HiddenState::LEN, GPT2_TOKEN_ROWS)?,
            mlp_up: DeviceBuffer::zeroed(stream, MlpActivation::LEN)?,
            mlp_down_input: RowwiseTapeBuffers::new(stream, MlpActivation::LEN, GPT2_TOKEN_ROWS)?,
        })
    }

    pub fn tape(&mut self) -> BlockForwardTape<'_> {
        BlockForwardTape {
            ln_1: self.ln_1.tape(),
            qkv_input_nvfp4: self.qkv_input.tape(),
            qkv: &mut self.qkv,
            attention_out: &mut self.attention_out,
            attention_log_sum_exp: &mut self.attention_log_sum_exp,
            c_proj_input_nvfp4: self.c_proj_input.tape(),
            ln_2: self.ln_2.tape(),
            mlp_up_input_nvfp4: self.mlp_up_input.tape(),
            mlp_up: &mut self.mlp_up,
            mlp_down_input_nvfp4: self.mlp_down_input.tape(),
        }
    }

    pub fn saved(&self, batch_size: u32, seq_len: u32, row_count: u32) -> BlockForwardSaved<'_> {
        BlockForwardSaved {
            batch_size,
            seq_len,
            row_count,
            ln_1: self.ln_1.saved(row_count),
            qkv_input_nvfp4: self.qkv_input.saved(),
            qkv: &self.qkv,
            attention_out: &self.attention_out,
            attention_log_sum_exp: &self.attention_log_sum_exp,
            c_proj_input_nvfp4: self.c_proj_input.saved(),
            ln_2: self.ln_2.saved(row_count),
            mlp_up_input_nvfp4: self.mlp_up_input.saved(),
            mlp_up: &self.mlp_up,
            mlp_down_input_nvfp4: self.mlp_down_input.saved(),
        }
    }

    pub fn qkv(&self) -> &DeviceBuffer<u16> {
        &self.qkv
    }
}
