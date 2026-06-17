use cuda_core::DriverError;
use rust_kernels_cuda::embedding::EmbeddingArgs;

use super::args::{HiddenStateDevice, TokenEmbeddingArgs};
use crate::TokenEmbedding;
use crate::random::InitRng;
use crate::types::{Nvfp4ShapeInit, TokenEmbeddingShape};

#[derive(Clone, Debug)]
pub struct EmbeddingWeights {
    pub wte: TokenEmbedding,
}

impl EmbeddingWeights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        Self {
            wte: TokenEmbeddingShape::smooth_tensor(rng),
        }
    }

    pub fn forward<'a>(
        &self,
        args: TokenEmbeddingArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let TokenEmbeddingArgs {
            module,
            stream,
            tokens,
            token_embedding,
            batch_size,
            seq_len,
            row_count,
            residual,
            normalized,
            normalized_amax,
            mean,
            inv_std,
        } = args;

        module.token_embedding(EmbeddingArgs {
            stream,
            tokens,
            token_embedding,
            residual: &mut *residual,
            hidden_len: row_count * TokenEmbedding::COLS as u32,
            embedding_dim: TokenEmbedding::COLS as u32,
        })?;

        Ok(HiddenStateDevice {
            stream,
            batch_size,
            seq_len,
            row_count,
            residual,
            normalized,
            normalized_amax,
            mean,
            inv_std,
        })
    }
}
