use gpt2_nvfp4::Gpt2BackwardWeights;

use crate::upload::UploadedModel;

pub(super) fn backward_weights(uploaded: &UploadedModel) -> Gpt2BackwardWeights<'_> {
    Gpt2BackwardWeights {
        lm_head_weight: uploaded.token_embedding.device(),
        ln_f: uploaded.ln_f.tensors(),
        block_ln_1: std::array::from_fn(|i| uploaded.blocks[i].ln_1.tensors()),
        block_ln_2: std::array::from_fn(|i| uploaded.blocks[i].ln_2.tensors()),
        attention: std::array::from_fn(|i| uploaded.blocks[i].attention_tensors()),
        mlp: std::array::from_fn(|i| uploaded.blocks[i].mlp_tensors()),
    }
}
