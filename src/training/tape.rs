use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    GPT2_CONTEXT_LEN, GPT2_N_LAYER, Gpt2ForwardSaved, Gpt2ForwardTape, HiddenState, Logits,
};

use super::tape_block::BlockTapeBuffers;
use super::tape_leaf::{LayerNormTapeBuffers, RowwiseTapeBuffers, zero};

pub struct ForwardTapeBuffers {
    embedding_residual: DeviceBuffer<f32>,
    blocks: [BlockTapeBuffers; GPT2_N_LAYER],
    final_norm: LayerNormTapeBuffers,
    lm_head_input: RowwiseTapeBuffers,
    logits: DeviceBuffer<f32>,
}

impl ForwardTapeBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            embedding_residual: zero(stream, HiddenState::LEN)?,
            blocks: block_array(|| BlockTapeBuffers::new(stream))?,
            final_norm: LayerNormTapeBuffers::new(stream)?,
            lm_head_input: RowwiseTapeBuffers::new(stream, HiddenState::LEN, GPT2_CONTEXT_LEN)?,
            logits: zero(stream, Logits::LEN)?,
        })
    }

    pub fn saved<'a>(&'a self, tokens: &'a DeviceBuffer<u32>) -> Gpt2ForwardSaved<'a> {
        Gpt2ForwardSaved {
            tokens,
            embedding_residual: &self.embedding_residual,
            blocks: std::array::from_fn(|i| self.blocks[i].saved()),
            final_norm: self.final_norm.saved(),
            lm_head_input_nvfp4: self.lm_head_input.saved(),
            logits: &self.logits,
        }
    }

    pub fn tape(&mut self) -> Gpt2ForwardTape<'_> {
        let blocks = self.blocks.as_mut_ptr();
        Gpt2ForwardTape {
            embedding_residual: &mut self.embedding_residual,
            blocks: std::array::from_fn(|i| unsafe { (&mut *blocks.add(i)).tape() }),
            final_norm: self.final_norm.tape(),
            lm_head_input_nvfp4: self.lm_head_input.tape(),
            logits: &mut self.logits,
        }
    }
}

fn block_array<F, T>(mut f: F) -> Result<[T; GPT2_N_LAYER], DriverError>
where
    F: FnMut() -> Result<T, DriverError>,
{
    let values = (0..GPT2_N_LAYER)
        .map(|_| f())
        .collect::<Result<Vec<_>, _>>()?;
    match values.try_into() {
        Ok(array) => Ok(array),
        Err(_) => unreachable!("block array length must match GPT2_N_LAYER"),
    }
}
