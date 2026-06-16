pub trait TransformerKernelConfig {
    const VOCAB_SIZE: u32;
    const CONTEXT_LEN: u32;
    const EMBEDDING_DIM: u32;
    const HEAD_COUNT: u32;
    const MLP_DIM: u32;
    const QKV_DIM: u32;

    const HIDDEN_LEN: u32 = Self::CONTEXT_LEN * Self::EMBEDDING_DIM;
}
