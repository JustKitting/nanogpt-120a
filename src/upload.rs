mod tensor;

use cuda_core::CudaStream;
use gpt2_nvfp4::{
    AttentionProjectionTensors, Gpt2BackwardWeights, Gpt2BlockWeights, Gpt2Weights, MlpDownTensors,
    MlpProjectionTensors, MlpUpTensors, NextLatWeights,
};

use crate::AppResult;

use self::tensor::{upload_linear, upload_nvfp4};
pub use self::tensor::{UploadedLayerNorm, UploadedLinear, UploadedNvfp4};

pub struct UploadedModel {
    pub token_embedding: UploadedNvfp4,
    pub blocks: Vec<UploadedBlock>,
    pub ln_f: UploadedLayerNorm,
    pub next_latent: UploadedNextLat,
}

impl UploadedModel {
    pub fn new(stream: &CudaStream, weights: &Gpt2Weights) -> AppResult<Self> {
        Ok(Self {
            token_embedding: upload_nvfp4(stream, &weights.embeddings.wte)?,
            blocks: weights
                .h
                .iter()
                .map(|block| UploadedBlock::new(stream, block))
                .collect::<AppResult<_>>()?,
            ln_f: UploadedLayerNorm::new(stream, &weights.ln_f)?,
            next_latent: UploadedNextLat::new(stream, &weights.next_latent)?,
        })
    }

    pub fn backward_weights(&self) -> Gpt2BackwardWeights<'_> {
        Gpt2BackwardWeights {
            lm_head_weight: self.token_embedding.device(),
            ln_f: self.ln_f.tensors(),
            block_ln_1: std::array::from_fn(|i| self.blocks[i].ln_1.tensors()),
            block_ln_2: std::array::from_fn(|i| self.blocks[i].ln_2.tensors()),
            attention: std::array::from_fn(|i| self.blocks[i].attention_tensors()),
            mlp: std::array::from_fn(|i| self.blocks[i].mlp_tensors()),
        }
    }
}

pub struct UploadedNextLat {
    pub norm: UploadedLayerNorm,
    pub input_projection: UploadedLinear,
    pub transition: UploadedLinear,
    pub output_projection: UploadedLinear,
}

impl UploadedNextLat {
    fn new(stream: &CudaStream, weights: &NextLatWeights) -> AppResult<Self> {
        Ok(Self {
            norm: UploadedLayerNorm {
                weight: upload_nvfp4(stream, &weights.norm_weight)?,
                bias: upload_nvfp4(stream, &weights.norm_bias)?,
            },
            input_projection: upload_linear(stream, &weights.input_projection)?,
            transition: upload_linear(stream, &weights.transition)?,
            output_projection: upload_linear(stream, &weights.output_projection)?,
        })
    }
}

pub struct UploadedBlock {
    pub ln_1: UploadedLayerNorm,
    pub attn_qkv: UploadedLinear,
    pub attn_c_proj: UploadedLinear,
    pub ln_2: UploadedLayerNorm,
    pub mlp_up: UploadedLinear,
    pub mlp_down: UploadedLinear,
}

impl UploadedBlock {
    fn new(stream: &CudaStream, block: &Gpt2BlockWeights) -> AppResult<Self> {
        Ok(Self {
            ln_1: UploadedLayerNorm::new(stream, &block.ln_1)?,
            attn_qkv: upload_linear(stream, &block.attn.c_attn)?,
            attn_c_proj: upload_linear(stream, &block.attn.c_proj)?,
            ln_2: UploadedLayerNorm::new(stream, &block.ln_2)?,
            mlp_up: upload_linear(stream, &block.mlp.c_fc)?,
            mlp_down: upload_linear(stream, &block.mlp.c_proj)?,
        })
    }

    pub fn attention_tensors(&self) -> AttentionProjectionTensors<'_> {
        AttentionProjectionTensors {
            qkv_weight: self.attn_qkv.weight.mma(),
            qkv_bias: self.attn_qkv.bias.device(),
            c_proj_weight: self.attn_c_proj.weight.mma(),
            c_proj_bias: self.attn_c_proj.bias.device(),
        }
    }

    pub fn mlp_tensors(&self) -> MlpProjectionTensors<'_> {
        MlpProjectionTensors {
            up: MlpUpTensors {
                weight: self.mlp_up.weight.mma(),
                bias: self.mlp_up.bias.device(),
            },
            down: MlpDownTensors {
                weight: self.mlp_down.weight.mma(),
                bias: self.mlp_down.bias.device(),
            },
        }
    }
}
