mod tensor;

pub use tensor::Nvfp4Tensor;

pub const GPT2_VOCAB_SIZE: usize = 50_257;
pub const GPT2_CONTEXT_LEN: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gpt2Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
}

impl Gpt2Config {
    pub const fn gpt2_124m() -> Self {
        Self {
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
        }
    }

    pub const fn vocab_size(self) -> usize {
        GPT2_VOCAB_SIZE
    }

    pub const fn max_seq_len(self) -> usize {
        GPT2_CONTEXT_LEN
    }

    pub const fn head_dim(self) -> usize {
        self.n_embd / self.n_head
    }

    pub const fn mlp_hidden(self) -> usize {
        4 * self.n_embd
    }
}

#[derive(Clone, Debug)]
pub struct Gpt2Weights {
    pub config: Gpt2Config,
    pub wte: Nvfp4Tensor,
    pub wpe: Nvfp4Tensor,
    pub h: Vec<Gpt2BlockWeights>,
    pub ln_f: LayerNormWeights,
}

#[derive(Clone, Debug)]
pub struct Gpt2BlockWeights {
    pub ln_1: LayerNormWeights,
    pub attn: AttentionWeights,
    pub ln_2: LayerNormWeights,
    pub mlp: MlpWeights,
}

#[derive(Clone, Debug)]
pub struct AttentionWeights {
    pub c_attn: LinearWeights,
    pub c_proj: LinearWeights,
}

#[derive(Clone, Debug)]
pub struct MlpWeights {
    pub c_fc: LinearWeights,
    pub c_proj: LinearWeights,
}

#[derive(Clone, Debug)]
pub struct LinearWeights {
    pub weight: Nvfp4Tensor,
    pub bias: Nvfp4Tensor,
}

#[derive(Clone, Debug)]
pub struct LayerNormWeights {
    pub weight: Nvfp4Tensor,
    pub bias: Nvfp4Tensor,
}
