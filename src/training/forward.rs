use gpt2_nvfp4::{
    GPT2_VOCAB_SIZE, Gpt2ForwardArgs, HiddenStateNvfp4, MlpActivationNvfp4, MlpDownTensors,
    MlpUpTensors, TokenEmbeddingArgs,
};

use super::{OptimizerTrace, TokenBatch, TrainStats, Trainer};
use crate::AppResult;

impl Trainer {
    pub fn forward_step(&mut self, batch: &TokenBatch) -> AppResult<TrainStats> {
        let stream = self.runtime.stream.as_ref();
        let buffers = &mut self.buffers;
        let uploaded = &self.uploaded;

        self.model.forward(Gpt2ForwardArgs {
            embeddings: TokenEmbeddingArgs {
                module: &self.runtime.embedding,
                stream,
                tokens: &batch.tokens,
                token_embedding: uploaded.token_embedding.device(),
                batch_size: batch.batch_size as u32,
                seq_len: batch.seq_len as u32,
                row_count: batch.token_count as u32,
                residual: &mut buffers.residual,
                normalized: &mut buffers.normalized,
                normalized_amax: &mut buffers.normalized_amax,
                mean: &mut buffers.mean,
                inv_std: &mut buffers.inv_std,
            },
            attention_module: &self.runtime.attention,
            quant_module: &self.runtime.quant,
            layer_norm_module: &self.runtime.layer_norm,
            mlp_module: &self.runtime.mlp,
            lm_head_module: &self.runtime.lm_head,
            hidden_nvfp4: HiddenStateNvfp4 {
                bytes: &mut buffers.hidden_bytes,
                scales: &mut buffers.hidden_scales,
                global_scales: &mut buffers.hidden_globals,
            },
            mlp_activation_nvfp4: MlpActivationNvfp4 {
                bytes: &mut buffers.mlp_bytes,
                scales: &mut buffers.mlp_scales,
                global_scales: &mut buffers.mlp_globals,
            },
            attention_qkv_weights: std::array::from_fn(|i| {
                uploaded.blocks[i].attn_qkv.weight.mma()
            }),
            attention_qkv_biases: std::array::from_fn(|i| {
                uploaded.blocks[i].attn_qkv.bias.device()
            }),
            attention_c_proj_weights: std::array::from_fn(|i| {
                uploaded.blocks[i].attn_c_proj.weight.mma()
            }),
            attention_c_proj_biases: std::array::from_fn(|i| {
                uploaded.blocks[i].attn_c_proj.bias.device()
            }),
            block_ln_1: std::array::from_fn(|i| uploaded.blocks[i].ln_1.tensors()),
            block_ln_2: std::array::from_fn(|i| uploaded.blocks[i].ln_2.tensors()),
            mlp_up: std::array::from_fn(|i| MlpUpTensors {
                weight: uploaded.blocks[i].mlp_up.weight.mma(),
                bias: uploaded.blocks[i].mlp_up.bias.device(),
            }),
            mlp_down: std::array::from_fn(|i| MlpDownTensors {
                weight: uploaded.blocks[i].mlp_down.weight.mma(),
                bias: uploaded.blocks[i].mlp_down.bias.device(),
            }),
            ln_f: uploaded.ln_f.tensors(),
            attention_qkv: &mut buffers.qkv,
            attention_log_sum_exp: &mut buffers.log_sum_exp,
            mlp_pre_activation: &mut buffers.mlp_pre,
            mlp_activation: &mut buffers.mlp_act,
            logits: &mut buffers.logits,
            tape: Some(buffers.tape.tape()),
        })?;

        Ok(TrainStats {
            tokens: batch.token_count,
            logits: batch.token_count * GPT2_VOCAB_SIZE,
            finite: true,
            nonzero: false,
            loss: 0.0,
            forward_ms: 0.0,
            backward_enqueue_ms: 0.0,
            loss_sync_ms: 0.0,
            optimizer_ms: 0.0,
            optimizer: OptimizerTrace::default(),
            diagnostics: None,
        })
    }
}
