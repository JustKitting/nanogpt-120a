use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::device_copy::copy_device;
use super::types::{BlockForwardTape, Gpt2ForwardTape};
use crate::types::Gpt2ForwardSaved;

impl<'a> Gpt2ForwardTape<'a> {
    pub fn saved<'t>(&'t self, tokens: &'t DeviceBuffer<u32>) -> Gpt2ForwardSaved<'t> {
        Gpt2ForwardSaved {
            tokens,
            embedding_residual: &*self.embedding_residual,
            blocks: std::array::from_fn(|index| self.blocks[index].saved()),
            final_norm: self.final_norm.saved(),
            logits: &*self.logits,
        }
    }

    pub(crate) fn block(&mut self, index: usize) -> BlockForwardTape<'_> {
        self.blocks[index].reborrow()
    }

    pub(crate) fn save_embedding(
        &mut self,
        stream: &CudaStream,
        residual: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, residual, self.embedding_residual)
    }

    pub(crate) fn save_logits(
        &mut self,
        stream: &CudaStream,
        logits: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, logits, self.logits)
    }
}
