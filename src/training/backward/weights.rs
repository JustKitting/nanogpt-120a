use gpt2_nvfp4::{
    AttentionProjectionTensors, Gpt2BackwardWeights, MlpDownTensors, MlpProjectionTensors,
    MlpUpTensors,
};

use crate::upload::UploadedModel;

pub(super) fn backward_weights(uploaded: &UploadedModel) -> Gpt2BackwardWeights<'_> {
    Gpt2BackwardWeights {
        lm_head_weight: uploaded.token_embedding.device(),
        ln_f: uploaded.ln_f.tensors(),
        block_ln_1: std::array::from_fn(|i| uploaded.blocks[i].ln_1.tensors()),
        block_ln_2: std::array::from_fn(|i| uploaded.blocks[i].ln_2.tensors()),
        attention: std::array::from_fn(|i| attention_weights(uploaded, i)),
        mlp: std::array::from_fn(|i| mlp_weights(uploaded, i)),
    }
}

fn attention_weights(uploaded: &UploadedModel, i: usize) -> AttentionProjectionTensors<'_> {
    AttentionProjectionTensors {
        qkv_weight: uploaded.blocks[i].attn_qkv.weight.mma(),
        qkv_bias: uploaded.blocks[i].attn_qkv.bias.device(),
        c_proj_weight: uploaded.blocks[i].attn_c_proj.weight.mma(),
        c_proj_bias: uploaded.blocks[i].attn_c_proj.bias.device(),
    }
}

fn mlp_weights(uploaded: &UploadedModel, i: usize) -> MlpProjectionTensors<'_> {
    MlpProjectionTensors {
        up: MlpUpTensors {
            weight: uploaded.blocks[i].mlp_up.weight.mma(),
            bias: uploaded.blocks[i].mlp_up.bias.device(),
        },
        down: MlpDownTensors {
            weight: uploaded.blocks[i].mlp_down.weight.mma(),
            bias: uploaded.blocks[i].mlp_down.bias.device(),
        },
    }
}
