use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_N_LAYER, GPT2_TOKEN_ROWS, Gpt2ForwardSaved, Gpt2ForwardTape, HiddenState};

use super::tape_block::BlockTapeBuffers;
use super::tape_leaf::{LayerNormTapeBuffers, RowwiseTapeBuffers};

pub struct ForwardTapeBuffers {
    blocks: [BlockTapeBuffers; GPT2_N_LAYER],
    final_norm: LayerNormTapeBuffers,
    lm_head_input: RowwiseTapeBuffers,
}

impl ForwardTapeBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            blocks: block_array(|| BlockTapeBuffers::new(stream))?,
            final_norm: LayerNormTapeBuffers::new(stream)?,
            lm_head_input: RowwiseTapeBuffers::new(stream, HiddenState::LEN, GPT2_TOKEN_ROWS)?,
        })
    }

    pub fn saved<'a>(
        &'a self,
        tokens: &'a DeviceBuffer<u32>,
        batch_size: u32,
        seq_len: u32,
        row_count: u32,
        logits: &'a DeviceBuffer<f32>,
    ) -> Gpt2ForwardSaved<'a> {
        Gpt2ForwardSaved {
            tokens,
            batch_size,
            seq_len,
            row_count,
            blocks: std::array::from_fn(|i| self.blocks[i].saved(batch_size, seq_len, row_count)),
            final_norm: self.final_norm.saved(row_count),
            lm_head_input_nvfp4: self.lm_head_input.saved(),
            logits,
        }
    }

    pub fn tape(&mut self) -> Gpt2ForwardTape<'_> {
        let blocks = self.blocks.as_mut_ptr();
        Gpt2ForwardTape {
            blocks: std::array::from_fn(|i| unsafe { (&mut *blocks.add(i)).tape() }),
            final_norm: self.final_norm.tape(),
            lm_head_input_nvfp4: self.lm_head_input.tape(),
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
