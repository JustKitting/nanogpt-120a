use crate::training::TrainStats;

pub fn log_diagnostics(step: usize, stats: &TrainStats) {
    let Some(trace) = &stats.diagnostics else {
        return;
    };

    println!(
        "trace step={step} updates={} positive_update_dot={} zero_grad_changed={} max_update_to_weight_rms={:.6e} dlogits_rms={:.6e} dlogits_max={:.6e} d_lm_head_rms={:.6e} d_lm_head_max={:.6e} d_embedding_rms={:.6e} d_embedding_max={:.6e} token_embedding_global_before={:.6e} token_embedding_global_after={:.6e} token_embedding_changed_bytes={} token_embedding_hash_before={:016x} token_embedding_hash_after={:016x}",
        trace.update_count,
        trace.positive_update_dot_count,
        trace.zero_grad_changed_count,
        trace.max_update_to_weight_rms,
        trace.dlogits_rms,
        trace.dlogits_max,
        trace.d_lm_head_rms,
        trace.d_lm_head_max,
        trace.d_embedding_rms,
        trace.d_embedding_max,
        trace.token_embedding_global_before,
        trace.token_embedding_global_after,
        trace.token_embedding_changed_bytes,
        trace.token_embedding_hash_before,
        trace.token_embedding_hash_after,
    );

    for update in &trace.updates {
        println!(
            "update step={step} tensor={} len={} grad_rms={:.6e} grad_max={:.6e} grad_nonzero={} grad_finite={} weight_rms_before={:.6e} weight_rms_after={:.6e} delta_rms={:.6e} delta_max={:.6e} update_to_weight_rms={:.6e} delta_grad_dot={:.6e} delta_grad_cos={:.6e} predicted_delta_rms={:.6e} predicted_delta_grad_dot={:.6e} predicted_delta_grad_cos={:.6e} quant_error_rms={:.6e} quant_error_to_predicted_delta_rms={:.6e} changed_bytes={} changed_scales={} global_before={:.6e} global_after={:.6e}",
            update.name,
            update.len,
            update.grad_rms,
            update.grad_max,
            update.grad_nonzero,
            update.grad_finite,
            update.weight_rms_before,
            update.weight_rms_after,
            update.delta_rms,
            update.delta_max,
            update.update_to_weight_rms,
            update.delta_grad_dot,
            update.delta_grad_cos,
            update.predicted_delta_rms,
            update.predicted_delta_grad_dot,
            update.predicted_delta_grad_cos,
            update.quant_error_rms,
            update.quant_error_to_predicted_delta_rms,
            update.changed_bytes,
            update.changed_scales,
            update.global_before,
            update.global_after,
        );
    }
}
