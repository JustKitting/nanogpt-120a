use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::embedding::{EmbeddingArgs, EmbeddingModule, Nvfp4DeviceTensor};

use crate::random::InitRng;
use crate::{GPT2_RMS_NORM_EPSILON, HiddenState, TokenEmbedding};

use super::{Nvfp4ShapeInit, TokenEmbeddingShape};

pub struct TokenEmbeddingArgs<'a> {
    pub module: &'a EmbeddingModule,
    pub stream: &'a CudaStream,
    pub tokens: &'a DeviceBuffer<u32>,
    pub token_embedding: Nvfp4DeviceTensor<'a>,
    pub rms_weight: Nvfp4DeviceTensor<'a>,
    pub hidden: &'a mut DeviceBuffer<f32>,
}

pub struct HiddenStateDevice<'a> {
    pub stream: &'a CudaStream,
    pub hidden: &'a mut DeviceBuffer<f32>,
}

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
            rms_weight,
            hidden,
        } = args;

        module.token_embedding_rmsnorm(EmbeddingArgs {
            stream,
            tokens,
            token_embedding,
            rms_weight,
            hidden: &mut *hidden,
            hidden_len: HiddenState::LEN as u32,
            embedding_dim: TokenEmbedding::COLS as u32,
            epsilon: GPT2_RMS_NORM_EPSILON,
        })?;

        Ok(HiddenStateDevice { stream, hidden })
    }
}
