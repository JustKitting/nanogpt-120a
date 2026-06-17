use gpt2_nvfp4::{
    AttentionProjectionTensors, Gpt2BackwardArgs, Gpt2BackwardSeeds, Gpt2BackwardWeights,
    MlpDownTensors, MlpProjectionTensors, MlpUpTensors, gpt2_backward,
};
use std::time::Instant;

use super::{TokenBatch, TrainStats, Trainer};
use crate::AppResult;

impl Trainer {
    pub fn train_step(&mut self, batch: &TokenBatch) -> AppResult<TrainStats> {
        let forward_start = Instant::now();
        let mut stats = self.forward_step(batch)?;
        stats.forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;
        let stream = self.runtime.stream.as_ref();

        let backward_start = Instant::now();
        {
            let saved = self.buffers.tape.saved(&batch.tokens);
            let weights = backward_weights(&self.uploaded);
            let mut backward = self.buffers.backward.parts();
            super::grad_clear::clear_backward_parts(stream, &mut backward)?;

            gpt2_backward(Gpt2BackwardArgs {
                stream,
                modules: self.runtime.backward_modules(),
                saved,
                weights,
                targets: &batch.targets,
                losses: backward.losses,
                d_lm_head_weight: backward.d_lm_head_weight,
                grads: backward.grads,
                scratch: self.buffers.scratch.scratch(),
                seeds: Gpt2BackwardSeeds::from_rng(&mut self.rng),
            })?;
        }
        stats.backward_ms = backward_start.elapsed().as_secs_f64() * 1000.0;

        let loss_sync_start = Instant::now();
        let losses = self.buffers.backward.losses.to_host_vec(stream)?;
        stats.loss = losses.iter().sum::<f32>() / losses.len() as f32;
        stats.finite &= losses.iter().all(|value| value.is_finite());
        stats.nonzero |= losses.iter().any(|value| value.abs() > 0.0);
        stats.loss_sync_ms = loss_sync_start.elapsed().as_secs_f64() * 1000.0;

        let optimizer_start = Instant::now();
        let updates = super::optimizer_apply::apply_weight_updates(
            stream,
            &self.runtime,
            batch,
            &mut self.uploaded,
            &mut self.buffers.backward,
            &mut self.buffers.optimizer,
            &mut self.buffers.optimizer_state,
            &mut self.buffers.aurora,
        )?;
        stats.optimizer = updates.trace;
        stats.optimizer_ms = optimizer_start.elapsed().as_secs_f64() * 1000.0;
        stats.diagnostics = updates.diagnostics;

        Ok(stats)
    }
}

fn backward_weights(uploaded: &crate::upload::UploadedModel) -> Gpt2BackwardWeights<'_> {
    Gpt2BackwardWeights {
        lm_head_weight: uploaded.token_embedding.device(),
        ln_f: uploaded.ln_f.tensors(),
        block_ln_1: std::array::from_fn(|i| uploaded.blocks[i].ln_1.tensors()),
        block_ln_2: std::array::from_fn(|i| uploaded.blocks[i].ln_2.tensors()),
        attention: std::array::from_fn(|i| attention_weights(uploaded, i)),
        mlp: std::array::from_fn(|i| mlp_weights(uploaded, i)),
    }
}

fn attention_weights(
    uploaded: &crate::upload::UploadedModel,
    i: usize,
) -> AttentionProjectionTensors<'_> {
    AttentionProjectionTensors {
        qkv_weight: uploaded.blocks[i].attn_qkv.weight.mma(),
        qkv_bias: uploaded.blocks[i].attn_qkv.bias.device(),
        c_proj_weight: uploaded.blocks[i].attn_c_proj.weight.mma(),
        c_proj_bias: uploaded.blocks[i].attn_c_proj.bias.device(),
    }
}

fn mlp_weights(uploaded: &crate::upload::UploadedModel, i: usize) -> MlpProjectionTensors<'_> {
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
