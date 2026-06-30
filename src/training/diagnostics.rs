use cuda_core::{CudaStream, DeviceBuffer};

use crate::AppResult;
use crate::upload::UploadedModel;

use super::grads::BackwardBuffers;
use super::optimizer_state::OptimizerStateBuffers;

mod update;

use update::{
    PendingTensorUpdateDiagnostics, changed_bytes, collect_update_snapshots,
    finish_update_snapshots,
};

const TRAIN_TRACE_ENV: &str = "TRAIN_TRACE";

pub struct TrainingDiagnostics {
    pub update_count: usize,
    pub positive_update_dot_count: usize,
    pub zero_grad_changed_count: usize,
    pub max_update_to_weight_rms: f32,
    pub dlogits_rms: f32,
    pub dlogits_max: f32,
    pub d_lm_head_rms: f32,
    pub d_lm_head_max: f32,
    pub d_embedding_rms: f32,
    pub d_embedding_max: f32,
    pub token_embedding_global_before: f32,
    pub token_embedding_global_after: f32,
    pub token_embedding_changed_bytes: usize,
    pub token_embedding_hash_before: u64,
    pub token_embedding_hash_after: u64,
    pub updates: Vec<TensorUpdateDiagnostics>,
}

pub struct TensorUpdateDiagnostics {
    pub name: String,
    pub len: usize,
    pub grad_rms: f32,
    pub grad_max: f32,
    pub grad_nonzero: usize,
    pub grad_finite: bool,
    pub weight_rms_before: f32,
    pub weight_rms_after: f32,
    pub delta_rms: f32,
    pub delta_max: f32,
    pub update_to_weight_rms: f32,
    pub delta_grad_dot: f32,
    pub delta_grad_cos: f32,
    pub predicted_delta_rms: f32,
    pub predicted_delta_grad_dot: f32,
    pub predicted_delta_grad_cos: f32,
    pub quant_error_rms: f32,
    pub quant_error_to_predicted_delta_rms: f32,
    pub changed_bytes: usize,
    pub changed_scales: usize,
    pub global_before: f32,
    pub global_after: f32,
}

pub struct PendingTrainingDiagnostics {
    diagnostics: TrainingDiagnostics,
    token_embedding_bytes_before: Vec<u8>,
    updates: Vec<PendingTensorUpdateDiagnostics>,
}

impl PendingTrainingDiagnostics {
    pub fn collect(
        stream: &CudaStream,
        uploaded: &UploadedModel,
        grads: &BackwardBuffers,
        state: &OptimizerStateBuffers,
        step: u32,
        average_coefficient: f32,
    ) -> AppResult<Self> {
        let token_embedding_bytes_before = uploaded.token_embedding.bytes.to_host_vec(stream)?;
        let (dlogits_rms, dlogits_max) = f32_buffer_stats(stream, &grads.dlogits)?;
        let (d_lm_head_rms, d_lm_head_max) = f32_buffer_stats(stream, &grads.d_lm_head_weight)?;
        let (d_embedding_rms, d_embedding_max) =
            f32_buffer_stats(stream, &grads.d_embedding_residual)?;
        let updates =
            collect_update_snapshots(stream, uploaded, grads, state, step, average_coefficient)?;
        let token_embedding_global = uploaded.token_embedding.global_scale_to_host(stream)?;

        Ok(Self {
            diagnostics: TrainingDiagnostics {
                update_count: updates.len(),
                positive_update_dot_count: 0,
                zero_grad_changed_count: 0,
                max_update_to_weight_rms: 0.0,
                dlogits_rms,
                dlogits_max,
                d_lm_head_rms,
                d_lm_head_max,
                d_embedding_rms,
                d_embedding_max,
                token_embedding_global_before: token_embedding_global,
                token_embedding_global_after: token_embedding_global,
                token_embedding_changed_bytes: 0,
                token_embedding_hash_before: hash_bytes(&token_embedding_bytes_before),
                token_embedding_hash_after: 0,
                updates: Vec::new(),
            },
            token_embedding_bytes_before,
            updates,
        })
    }

    pub fn finish(
        mut self,
        stream: &CudaStream,
        uploaded: &UploadedModel,
    ) -> AppResult<TrainingDiagnostics> {
        let after = uploaded.token_embedding.bytes.to_host_vec(stream)?;
        self.diagnostics.token_embedding_global_after =
            uploaded.token_embedding.global_scale_to_host(stream)?;
        self.diagnostics.token_embedding_changed_bytes =
            changed_bytes(&self.token_embedding_bytes_before, &after);
        self.diagnostics.token_embedding_hash_after = hash_bytes(&after);
        self.diagnostics.updates = finish_update_snapshots(stream, uploaded, self.updates)?;
        self.diagnostics.positive_update_dot_count = self
            .diagnostics
            .updates
            .iter()
            .filter(|update| update.delta_grad_dot > 0.0)
            .count();
        self.diagnostics.zero_grad_changed_count = self
            .diagnostics
            .updates
            .iter()
            .filter(|update| update.grad_nonzero == 0 && update.changed_bytes > 0)
            .count();
        self.diagnostics.max_update_to_weight_rms = self
            .diagnostics
            .updates
            .iter()
            .map(|update| update.update_to_weight_rms)
            .fold(0.0, f32::max);
        Ok(self.diagnostics)
    }
}

pub fn enabled() -> bool {
    std::env::var(TRAIN_TRACE_ENV)
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn f32_buffer_stats(stream: &CudaStream, buffer: &DeviceBuffer<f32>) -> AppResult<(f32, f32)> {
    let values = buffer.to_host_vec(stream)?;
    let mut sum_sq = 0.0f64;
    let mut max = 0.0f32;

    for value in &values {
        let abs = value.abs();
        sum_sq += (*value as f64) * (*value as f64);
        max = max.max(abs);
    }

    Ok(((sum_sq / values.len() as f64).sqrt() as f32, max))
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
