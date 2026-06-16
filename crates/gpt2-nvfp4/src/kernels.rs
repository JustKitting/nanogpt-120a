use crate::{HiddenState, PositionEmbedding, TokenEmbedding, TokenIds};

pub(crate) fn launch_embedding_kernel(
    tokens: &TokenIds,
    token_embedding: &TokenEmbedding,
    position_embedding: &PositionEmbedding,
    hidden: &mut HiddenState,
) {
    // GPU math contract:
    //
    // token_embedding:    [GPT2_VOCAB_SIZE, GPT2_N_EMBD] in NVFP4
    // position_embedding: [GPT2_CONTEXT_LEN, GPT2_N_EMBD] in NVFP4
    // tokens:             [GPT2_CONTEXT_LEN] token ids
    // hidden:             [GPT2_CONTEXT_LEN, GPT2_N_EMBD] activation output
    //
    // One GPU thread maps to one hidden element:
    //   pos = linear_index / GPT2_N_EMBD
    //   dim = linear_index % GPT2_N_EMBD
    //
    // The kernel gathers the token row and adds the matching position row:
    //   hidden[pos, dim] = token_embedding[tokens[pos], dim]
    //                    + position_embedding[pos, dim]
    //
    // This is the GPT-2 first-stage embedding op: WTE[token] + WPE[pos] -> hidden.
    let _ = (tokens, token_embedding, position_embedding, hidden);
    unimplemented!("GPU embedding kernel launch is not wired yet");
}
