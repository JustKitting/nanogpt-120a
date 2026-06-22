use cuda_core::DeviceBuffer;

use super::types::{BlockForwardTape, Gpt2ForwardTape};
use crate::types::Gpt2ForwardSaved;

impl<'a> Gpt2ForwardTape<'a> {
    pub fn saved<'t>(
        &'t self,
        tokens: &'t DeviceBuffer<u32>,
        logits: &'t DeviceBuffer<f32>,
        batch_size: u32,
        seq_len: u32,
        row_count: u32,
    ) -> Gpt2ForwardSaved<'t> {
        Gpt2ForwardSaved {
            tokens,
            batch_size,
            seq_len,
            row_count,
            blocks: std::array::from_fn(|index| {
                self.blocks[index].saved(batch_size, seq_len, row_count)
            }),
            final_norm: self.final_norm.saved(row_count),
            lm_head_input_nvfp4: self.lm_head_input_nvfp4.saved(),
            logits,
        }
    }

    pub(crate) fn block(&mut self, index: usize) -> BlockForwardTape<'_> {
        self.blocks[index].reborrow()
    }
}
