use std::marker::PhantomData;

use crate::{GPT2_MLP, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_QKV, GPT2_TOKEN_ROWS, GPT2_VOCAB_SIZE};

pub trait BufferShape {
    const LEN: usize;
}

#[derive(Clone, Debug)]
pub struct F32Buffer<S: BufferShape> {
    shape: PhantomData<S>,
}

impl<S: BufferShape> F32Buffer<S> {
    pub const LEN: usize = S::LEN;
}

macro_rules! buffer_shape {
    ($name:ident, $len:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub enum $name {}

        impl BufferShape for $name {
            const LEN: usize = $len;
        }
    };
}

buffer_shape!(HiddenStateShape, GPT2_TOKEN_ROWS * GPT2_N_EMBD);
buffer_shape!(QkvActivationShape, GPT2_TOKEN_ROWS * GPT2_QKV);
buffer_shape!(AttentionLogSumExpShape, GPT2_TOKEN_ROWS * GPT2_N_HEAD);
buffer_shape!(MlpActivationShape, GPT2_TOKEN_ROWS * GPT2_MLP);
buffer_shape!(LogitsShape, GPT2_TOKEN_ROWS * GPT2_VOCAB_SIZE);

pub type HiddenState = F32Buffer<HiddenStateShape>;
pub type QkvActivation = F32Buffer<QkvActivationShape>;
pub type AttentionLogSumExp = F32Buffer<AttentionLogSumExpShape>;
pub type MlpActivation = F32Buffer<MlpActivationShape>;
pub type Logits = F32Buffer<LogitsShape>;
