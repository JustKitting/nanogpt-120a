use crate::{
    GPT2_CONTEXT_LEN, GPT2_MLP, GPT2_N_EMBD, GPT2_QKV, GPT2_VOCAB_SIZE, Nvfp4Shape, Nvfp4Tensor,
};

use super::LinearWeights;

macro_rules! nvfp4_shape {
    ($name:ident, $rows:expr, $cols:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub enum $name {}

        impl Nvfp4Shape for $name {
            const ROWS: usize = $rows;
            const COLS: usize = $cols;
            const BYTE_LEN: usize = nvfp4_bytes($rows, $cols);
            const SCALE_LEN: usize = nvfp4_scales($rows, $cols);

            type Bytes = [u8; { nvfp4_bytes($rows, $cols) }];
            type Scales = [u8; { nvfp4_scales($rows, $cols) }];
        }
    };
}

nvfp4_shape!(TokenEmbeddingShape, GPT2_VOCAB_SIZE, GPT2_N_EMBD);
nvfp4_shape!(PositionEmbeddingShape, GPT2_CONTEXT_LEN, GPT2_N_EMBD);
nvfp4_shape!(HiddenVectorShape, 1, GPT2_N_EMBD);
nvfp4_shape!(QkvWeightShape, GPT2_N_EMBD, GPT2_QKV);
nvfp4_shape!(QkvVectorShape, 1, GPT2_QKV);
nvfp4_shape!(ResidualWeightShape, GPT2_N_EMBD, GPT2_N_EMBD);
nvfp4_shape!(MlpUpWeightShape, GPT2_N_EMBD, GPT2_MLP);
nvfp4_shape!(MlpVectorShape, 1, GPT2_MLP);
nvfp4_shape!(MlpDownWeightShape, GPT2_MLP, GPT2_N_EMBD);

pub type TokenEmbedding = Nvfp4Tensor<TokenEmbeddingShape>;
pub type PositionEmbedding = Nvfp4Tensor<PositionEmbeddingShape>;
pub type LayerNormTensor = Nvfp4Tensor<HiddenVectorShape>;
pub type QkvLinear = LinearWeights<QkvWeightShape, QkvVectorShape>;
pub type ResidualLinear = LinearWeights<ResidualWeightShape, HiddenVectorShape>;
pub type MlpUpLinear = LinearWeights<MlpUpWeightShape, MlpVectorShape>;
pub type MlpDownLinear = LinearWeights<MlpDownWeightShape, HiddenVectorShape>;

pub const fn nvfp4_bytes(rows: usize, cols: usize) -> usize {
    (rows * cols).div_ceil(2)
}

pub const fn nvfp4_scales(rows: usize, cols: usize) -> usize {
    (rows * cols).div_ceil(16)
}
