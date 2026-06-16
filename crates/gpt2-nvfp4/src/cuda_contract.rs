use crate::{
    GPT2_CONTEXT_LEN, GPT2_MLP, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_QKV, GPT2_VOCAB_SIZE, HiddenState,
    PositionEmbedding, QkvActivation, TokenEmbedding, TokenIds,
};
use rust_kernels_cuda::kernel_config::TransformerKernelConfig;

pub struct Gpt2KernelConfig;

impl TransformerKernelConfig for Gpt2KernelConfig {
    const VOCAB_SIZE: u32 = GPT2_VOCAB_SIZE as u32;
    const CONTEXT_LEN: u32 = GPT2_CONTEXT_LEN as u32;
    const EMBEDDING_DIM: u32 = GPT2_N_EMBD as u32;
    const HEAD_COUNT: u32 = GPT2_N_HEAD as u32;
    const MLP_DIM: u32 = GPT2_MLP as u32;
    const QKV_DIM: u32 = GPT2_QKV as u32;
}

const _: [(); GPT2_CONTEXT_LEN] = [(); TokenIds::LEN];
const _: [(); GPT2_CONTEXT_LEN] = [(); PositionEmbedding::ROWS];
const _: [(); GPT2_N_EMBD] = [(); TokenEmbedding::COLS];
const _: [(); GPT2_N_EMBD] = [(); PositionEmbedding::COLS];
const _: [(); GPT2_CONTEXT_LEN * GPT2_N_EMBD] = [(); HiddenState::LEN];
const _: [(); GPT2_CONTEXT_LEN * GPT2_QKV] = [(); QkvActivation::LEN];
