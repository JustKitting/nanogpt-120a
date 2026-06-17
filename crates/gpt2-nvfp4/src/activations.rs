use std::marker::PhantomData;

use crate::{GPT2_CONTEXT_LEN, GPT2_MLP, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_QKV, GPT2_VOCAB_SIZE};

pub trait BufferShape<T> {
    const LEN: usize;
    type Data: AsRef<[T]> + AsMut<[T]> + Clone;
}

#[derive(Clone, Debug)]
pub struct F32Buffer<S: BufferShape<f32>> {
    pub data: S::Data,
    shape: PhantomData<S>,
}

#[derive(Clone, Debug)]
pub struct U32Buffer<S: BufferShape<u32>> {
    pub data: S::Data,
    shape: PhantomData<S>,
}

impl<S: BufferShape<f32>> F32Buffer<S> {
    pub const LEN: usize = S::LEN;

    pub fn new(data: S::Data) -> Self {
        Self {
            data,
            shape: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        Self::LEN
    }

    pub fn is_empty(&self) -> bool {
        Self::LEN == 0
    }
}

impl<S: BufferShape<u32>> U32Buffer<S> {
    pub const LEN: usize = S::LEN;

    pub fn new(data: S::Data) -> Self {
        Self {
            data,
            shape: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        Self::LEN
    }

    pub fn is_empty(&self) -> bool {
        Self::LEN == 0
    }
}

macro_rules! buffer_shape {
    ($name:ident, $ty:ty, $len:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub enum $name {}

        impl BufferShape<$ty> for $name {
            const LEN: usize = $len;
            type Data = [$ty; $len];
        }
    };
}

buffer_shape!(TokenIdsShape, u32, GPT2_CONTEXT_LEN);
buffer_shape!(HiddenStateShape, f32, GPT2_CONTEXT_LEN * GPT2_N_EMBD);
buffer_shape!(QkvActivationShape, f32, GPT2_CONTEXT_LEN * GPT2_QKV);
buffer_shape!(
    AttentionScoresShape,
    f32,
    GPT2_N_HEAD * GPT2_CONTEXT_LEN * GPT2_CONTEXT_LEN
);
buffer_shape!(AttentionLseShape, f32, GPT2_N_HEAD * GPT2_CONTEXT_LEN);
buffer_shape!(MlpActivationShape, f32, GPT2_CONTEXT_LEN * GPT2_MLP);
buffer_shape!(LogitsShape, f32, GPT2_CONTEXT_LEN * GPT2_VOCAB_SIZE);

pub type TokenIds = U32Buffer<TokenIdsShape>;
pub type HiddenState = F32Buffer<HiddenStateShape>;
pub type QkvActivation = F32Buffer<QkvActivationShape>;
pub type AttentionScores = F32Buffer<AttentionScoresShape>;
pub type AttentionLse = F32Buffer<AttentionLseShape>;
pub type MlpActivation = F32Buffer<MlpActivationShape>;
pub type Logits = F32Buffer<LogitsShape>;
