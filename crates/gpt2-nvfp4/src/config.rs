pub const GPT2_VOCAB_SIZE: usize = llama2_tokenizer::VOCAB_SIZE;
include!(concat!(env!("OUT_DIR"), "/gpt2_shape.rs"));

pub const GPT2_TOKEN_ROWS: usize = GPT2_BATCH_SIZE * GPT2_SEQ_LEN;
pub const GPT2_CONTEXT_LEN: usize = GPT2_SEQ_LEN;

pub const GPT2_LAYER_NORM_EPSILON: f32 = 1.0e-5;

pub const GPT2_MLP: usize = 4 * GPT2_N_EMBD;
pub const GPT2_FULL_ATTENTION_QKV: usize = 3 * GPT2_N_EMBD;
pub const GPT2_KDA_ACTIVE_QKV: usize = 4 * GPT2_N_EMBD + GPT2_N_HEAD;
pub const GPT2_QKV: usize = align_kda_qkv(GPT2_KDA_ACTIVE_QKV);
pub const GPT2_Q_OFFSET: usize = 0;
pub const GPT2_K_OFFSET: usize = GPT2_N_EMBD;
pub const GPT2_V_OFFSET: usize = 2 * GPT2_N_EMBD;
pub const GPT2_KDA_G_OFFSET: usize = 3 * GPT2_N_EMBD;
pub const GPT2_KDA_BETA_OFFSET: usize = 4 * GPT2_N_EMBD;
pub const KDA_CHUNK_SIZE: usize = 64;
pub const KDA_DECAY_SCALE: f32 = 0.01;
pub const NEXTLAT_INPUT: usize = 2 * GPT2_N_EMBD;
pub const NEXTLAT_HIDDEN: usize = NEXTLAT_INPUT;

pub const KIMI_FULL_ATTENTION_PERIOD: usize = 4;

const fn align_up(value: usize, alignment: usize) -> usize {
    ((value + alignment - 1) / alignment) * alignment
}

const fn align_kda_qkv(value: usize) -> usize {
    let mut aligned = align_up(value, 32);
    while align_up(aligned, 64) % 128 != 0 {
        aligned += 32;
    }
    aligned
}

pub const fn uses_full_attention(block_index: usize) -> bool {
    block_index % KIMI_FULL_ATTENTION_PERIOD == KIMI_FULL_ATTENTION_PERIOD - 1
}

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
