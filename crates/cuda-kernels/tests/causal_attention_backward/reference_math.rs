use super::shape::*;

pub fn scores(qkv: &[f32], query: usize, head: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(query + 1);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    for key in 0..=query {
        let mut dot = 0.0;
        for dim in 0..HEAD_DIM {
            dot +=
                rope_value(qkv, query, head, dim, 0) * rope_value(qkv, key, head, dim, EMBEDDING);
        }
        values.push(dot * scale);
    }
    values
}

pub fn logsumexp(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max + values
        .iter()
        .map(|value| (*value - max).exp())
        .sum::<f32>()
        .ln()
}

pub fn backward(qkv: &[f32], out: &[f32], d_out: &[f32], log_sum_exp: &[f32]) -> Vec<f32> {
    let mut dq_rot = vec![0.0_f32; TOKEN_COUNT * EMBEDDING];
    let mut dk_rot = vec![0.0_f32; TOKEN_COUNT * EMBEDDING];
    let mut grad = vec![0.0_f32; TOKEN_COUNT * QKV_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    for query in 0..TOKEN_COUNT {
        for head in 0..HEADS {
            let row_d = softmax_d(out, d_out, query, head);
            let row_scores = scores(qkv, query, head);
            for key in 0..=query {
                let p = (row_scores[key] - log_sum_exp[log_sum_exp_index(query, head)]).exp();
                let ds = p * (d_out_dot_v(qkv, d_out, query, key, head) - row_d);
                for dim in 0..HEAD_DIM {
                    let h = hidden_index(query, head, dim);
                    let k = hidden_index(key, head, dim);
                    grad[qkv_index(key, head, dim, 2 * EMBEDDING)] +=
                        p * d_out[hidden_index(query, head, dim)];
                    dq_rot[h] += ds * rope_value(qkv, key, head, dim, EMBEDDING) * scale;
                    dk_rot[k] += ds * rope_value(qkv, query, head, dim, 0) * scale;
                }
            }
        }
    }
    apply_rope_backward(&mut grad, &dq_rot, &dk_rot);
    grad
}

fn softmax_d(out: &[f32], d_out: &[f32], query: usize, head: usize) -> f32 {
    let mut value = 0.0;
    for dim in 0..HEAD_DIM {
        let index = hidden_index(query, head, dim);
        value += out[index] * d_out[index];
    }
    value
}

fn d_out_dot_v(qkv: &[f32], d_out: &[f32], query: usize, key: usize, head: usize) -> f32 {
    let mut value = 0.0;
    for dim in 0..HEAD_DIM {
        value +=
            d_out[hidden_index(query, head, dim)] * qkv[qkv_index(key, head, dim, 2 * EMBEDDING)];
    }
    value
}

fn apply_rope_backward(grad: &mut [f32], dq_rot: &[f32], dk_rot: &[f32]) {
    for token in 0..TOKEN_COUNT {
        for head in 0..HEADS {
            for dim in 0..HEAD_DIM {
                let h = hidden_index(token, head, dim);
                grad[qkv_index(token, head, dim, 0)] = rope_raw_grad(
                    token,
                    dim,
                    dq_rot[h],
                    dq_rot[hidden_index(token, head, dim ^ 1)],
                );
                grad[qkv_index(token, head, dim, EMBEDDING)] = rope_raw_grad(
                    token,
                    dim,
                    dk_rot[h],
                    dk_rot[hidden_index(token, head, dim ^ 1)],
                );
            }
        }
    }
}

fn rope_value(qkv: &[f32], token: usize, head: usize, dim: usize, offset: usize) -> f32 {
    let value = qkv[qkv_index(token, head, dim, offset)];
    let paired = qkv[qkv_index(token, head, dim ^ 1, offset)];
    let (sin, cos) = rope_angle(token, dim).sin_cos();
    if dim & 1 == 0 {
        value * cos - paired * sin
    } else {
        paired * sin + value * cos
    }
}

fn rope_raw_grad(token: usize, dim: usize, grad_dim: f32, grad_pair: f32) -> f32 {
    let (sin, cos) = rope_angle(token, dim).sin_cos();
    if dim & 1 == 0 {
        grad_pair * sin + grad_dim * cos
    } else {
        -grad_pair * sin + grad_dim * cos
    }
}

fn rope_angle(token: usize, dim: usize) -> f32 {
    token as f32 * 10_000.0_f32.powf(-((dim & !1) as f32) / HEAD_DIM as f32)
}
