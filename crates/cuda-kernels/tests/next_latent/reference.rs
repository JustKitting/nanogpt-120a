use super::{BATCH_SIZE, EMBED, LAMBDA, ROW_COUNT, SEQ_LEN};

pub fn values(offset: f32) -> Vec<f32> {
    (0..ROW_COUNT * EMBED)
        .map(|index| offset + (index as f32 - 7.0) * 0.0625)
        .collect()
}

pub fn concat(next_token_embeddings: &[f32], current_states: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; ROW_COUNT * EMBED * 2];
    for row in 0..ROW_COUNT {
        for col in 0..EMBED {
            out[row * EMBED * 2 + col] = next_token_embeddings[row * EMBED + col];
            out[row * EMBED * 2 + EMBED + col] = current_states[row * EMBED + col];
        }
    }
    out
}

pub fn smooth_l1(predicted: &[f32], target: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut losses = vec![0.0_f32; ROW_COUNT];
    let mut grad = vec![0.0_f32; ROW_COUNT * EMBED];
    let grad_scale = LAMBDA / ((BATCH_SIZE * (SEQ_LEN - 1) * EMBED) as f32);

    for batch in 0..BATCH_SIZE {
        for pos in 0..SEQ_LEN - 1 {
            let row = batch * SEQ_LEN + pos;
            let mut local = 0.0;
            for col in 0..EMBED {
                let offset = row * EMBED + col;
                let target_offset = (row + 1) * EMBED + col;
                let diff = predicted[offset] - target[target_offset];
                let abs = diff.abs();
                let d = if abs < 1.0 {
                    local += 0.5 * diff * diff;
                    diff
                } else {
                    local += abs - 0.5;
                    diff.signum()
                };
                grad[offset] = d * grad_scale;
            }
            losses[row] = LAMBDA * local / EMBED as f32;
        }
    }

    (losses, grad)
}
