pub const GPT2_VOCAB_SIZE: usize = llama2_tokenizer::VOCAB_SIZE;
include!(concat!(env!("OUT_DIR"), "/gpt2_shape.rs"));

pub const GPT2_TOKEN_ROWS: usize = GPT2_BATCH_SIZE * GPT2_SEQ_LEN;
pub const GPT2_CONTEXT_LEN: usize = GPT2_SEQ_LEN;

pub const GPT2_LAYER_NORM_EPSILON: f32 = 1.0e-5;

pub const GPT2_MLP: usize = 4 * GPT2_N_EMBD;
pub const GPT2_QKV: usize = 3 * GPT2_N_EMBD;
pub const NEXTLAT_INPUT: usize = 2 * GPT2_N_EMBD;
pub const NEXTLAT_HIDDEN: usize = NEXTLAT_INPUT;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gpt2Config;

impl Gpt2Config {
    pub const fn gpt2_124m() -> Self {
        Self
    }

    pub const fn vocab_size(self) -> usize {
        GPT2_VOCAB_SIZE
    }

    pub const fn max_seq_len(self) -> usize {
        GPT2_SEQ_LEN
    }

    pub const fn head_dim(self) -> usize {
        GPT2_N_EMBD / GPT2_N_HEAD
    }

    pub const fn mlp_hidden(self) -> usize {
        GPT2_MLP
    }
}
