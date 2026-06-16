use crate::random::InitRng;
use crate::{GPT2_MLP, GPT2_N_EMBD, GPT2_QKV, GPT2_VOCAB_SIZE, Nvfp4Shape, Nvfp4Tensor};

use super::LinearWeights;
pub const fn nvfp4_bytes(rows: usize, cols: usize) -> usize {
    (rows * cols).div_ceil(2)
}

pub const fn nvfp4_scales(rows: usize, cols: usize) -> usize {
    (rows * cols).div_ceil(16)
}

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

            fn zero_bytes() -> Self::Bytes {
                [0; nvfp4_bytes($rows, $cols)]
            }

            fn zero_scales() -> Self::Scales {
                [0; nvfp4_scales($rows, $cols)]
            }
        }
    };
}

pub(crate) trait Nvfp4ShapeInit: Nvfp4Shape + Sized {
    fn smooth_tensor(rng: &mut InitRng) -> Nvfp4Tensor<Self> {
        let mut bytes = Self::zero_bytes();
        let mut scales = Self::zero_scales();
        fill_smooth_payload(bytes.as_mut(), rng);
        fill_neutral_scales(scales.as_mut());
        Nvfp4Tensor::new(bytes, scales, 0.02)
    }

    fn zero_tensor() -> Nvfp4Tensor<Self> {
        let bytes = Self::zero_bytes();
        let mut scales = Self::zero_scales();
        fill_neutral_scales(scales.as_mut());
        Nvfp4Tensor::new(bytes, scales, 1.0)
    }

    fn one_tensor() -> Nvfp4Tensor<Self> {
        let mut bytes = Self::zero_bytes();
        let mut scales = Self::zero_scales();
        bytes.as_mut().fill(pack_e2m1_pair(E2M1_ONE, E2M1_ONE));
        fill_neutral_scales(scales.as_mut());
        Nvfp4Tensor::new(bytes, scales, 1.0)
    }
}

impl<S: Nvfp4Shape> Nvfp4ShapeInit for S {}

fn fill_smooth_payload(bytes: &mut [u8], rng: &mut InitRng) {
    for byte in bytes {
        *byte = pack_e2m1_pair(smooth_e2m1(rng), smooth_e2m1(rng));
    }
}

fn fill_neutral_scales(scales: &mut [u8]) {
    scales.fill(E4M3_ONE);
}

fn smooth_e2m1(rng: &mut InitRng) -> u8 {
    let byte = rng.next_u8();
    if byte & 0x3 == 0 {
        E2M1_ZERO
    } else if byte & 0x4 == 0 {
        E2M1_POS_MIN
    } else {
        E2M1_NEG_MIN
    }
}

const E2M1_ZERO: u8 = 0x0;
const E2M1_POS_MIN: u8 = 0x1;
const E2M1_ONE: u8 = 0x2;
const E2M1_NEG_MIN: u8 = 0x9;
const E4M3_ONE: u8 = 0x38;

const fn pack_e2m1_pair(lo: u8, hi: u8) -> u8 {
    (lo & 0x0f) | ((hi & 0x0f) << 4)
}

nvfp4_shape!(TokenEmbeddingShape, GPT2_VOCAB_SIZE, GPT2_N_EMBD);
nvfp4_shape!(HiddenVectorShape, 1, GPT2_N_EMBD);
nvfp4_shape!(QkvWeightShape, GPT2_N_EMBD, GPT2_QKV);
nvfp4_shape!(QkvVectorShape, 1, GPT2_QKV);
nvfp4_shape!(ResidualWeightShape, GPT2_N_EMBD, GPT2_N_EMBD);
nvfp4_shape!(MlpUpWeightShape, GPT2_N_EMBD, GPT2_MLP);
nvfp4_shape!(MlpVectorShape, 1, GPT2_MLP);
nvfp4_shape!(MlpDownWeightShape, GPT2_MLP, GPT2_N_EMBD);

pub type TokenEmbedding = Nvfp4Tensor<TokenEmbeddingShape>;
pub type LayerNormTensor = Nvfp4Tensor<HiddenVectorShape>;
pub type QkvLinear = LinearWeights<QkvWeightShape, QkvVectorShape>;
pub type ResidualLinear = LinearWeights<ResidualWeightShape, HiddenVectorShape>;
pub type MlpUpLinear = LinearWeights<MlpUpWeightShape, MlpVectorShape>;
pub type MlpDownLinear = LinearWeights<MlpDownWeightShape, HiddenVectorShape>;
