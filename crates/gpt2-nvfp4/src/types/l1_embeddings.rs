use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::embedding::{EmbeddingArgs, EmbeddingModule, Nvfp4DeviceTensor};

use crate::Gpt2KernelConfig;
use crate::random::InitRng;

use super::{
    Nvfp4ShapeInit, PositionEmbedding, PositionEmbeddingShape, TokenEmbedding, TokenEmbeddingShape,
};

pub struct TokenPositionEmbeddingArgs<'a> {
    pub module: &'a EmbeddingModule,
    pub stream: &'a CudaStream,
    pub tokens: &'a DeviceBuffer<u32>,
    pub token_embedding: Nvfp4DeviceTensor<'a>,
    pub position_embedding: Nvfp4DeviceTensor<'a>,
    pub hidden: &'a mut DeviceBuffer<f32>,
}

pub struct HiddenStateDevice<'a> {
    pub stream: &'a CudaStream,
    pub hidden: &'a mut DeviceBuffer<f32>,
}

#[derive(Clone, Debug)]
pub struct EmbeddingWeights {
    pub wte: TokenEmbedding,
    pub wpe: PositionEmbedding,
}

impl EmbeddingWeights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        Self {
            wte: TokenEmbeddingShape::smooth_tensor(rng),
            wpe: PositionEmbeddingShape::smooth_tensor(rng),
        }
    }

    pub fn forward<'a>(
        &self,
        args: TokenPositionEmbeddingArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let TokenPositionEmbeddingArgs {
            module,
            stream,
            tokens,
            token_embedding,
            position_embedding,
            hidden,
        } = args;

        module.token_position_embedding::<Gpt2KernelConfig>(EmbeddingArgs::new(
            stream,
            tokens,
            token_embedding,
            position_embedding,
            &mut *hidden,
        ))?;

        Ok(HiddenStateDevice { stream, hidden })
    }
}
