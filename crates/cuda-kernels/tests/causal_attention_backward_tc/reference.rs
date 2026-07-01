use super::shape::*;

pub fn backward(
    qkv: &[f32],
    out: &[f32],
    d_out: &[f32],
    d_out_half: &[f32],
    log_sum_exp: &[f32],
) -> Vec<f32> {
    let mut dq_rot = vec![0.0_f32; TOKEN_COUNT * EMBEDDING];
    let mut dk_rot = vec![0.0_f32; TOKEN_COUNT * EMBEDDING];
    let mut grad = vec![0.0_f32; TOKEN_COUNT * QKV_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    for query in 0..TOKEN_COUNT {
        for head in 0..HEADS {
            let softmax_d = softmax_d(out, d_out, query, head);
            let row_scores = scores(qkv, query, head);
            for key in 0..=query {
                let p =
                    (row_scores[key] * scale - log_sum_exp[log_sum_exp_index(query, head)]).exp();
                let ds = p * (d_out_dot_v(qkv, d_out_half, query, key, head) - softmax_d);
                for dim in 0..HEAD_DIM {
                    let q_index = hidden_index(query, head, dim);
                    let k_index = hidden_index(key, head, dim);
                    grad[qkv_index(key, head, dim, 2 * EMBEDDING)] +=
                        p * d_out_half[hidden_index(query, head, dim)];
                    dq_rot[q_index] += ds * qkv_value(qkv, key, head, dim, EMBEDDING) * scale;
                    dk_rot[k_index] += ds * qkv_value(qkv, query, head, dim, 0) * scale;
                }
            }
        }
    }
    apply_rope_backward(&mut grad, &dq_rot, &dk_rot);
    grad
}

fn scores(qkv: &[f32], query: usize, head: usize) -> Vec<f32> {
    (0..=query)
        .map(|key| head_sum(|dim| qkv_value(qkv, query, head, dim, 0) * qkv_value(qkv, key, head, dim, EMBEDDING)))
        .collect()
}

fn softmax_d(out: &[f32], d_out: &[f32], query: usize, head: usize) -> f32 {
    head_sum(|dim| out[hidden_index(query, head, dim)] * d_out[hidden_index(query, head, dim)])
}

fn d_out_dot_v(qkv: &[f32], d_out: &[f32], query: usize, key: usize, head: usize) -> f32 {
    head_sum(|dim| d_out[hidden_index(query, head, dim)] * qkv[qkv_index(key, head, dim, 2 * EMBEDDING)])
}

fn head_sum(value: impl Fn(usize) -> f32) -> f32 {
    (0..HEAD_DIM).map(value).sum()
}

fn apply_rope_backward(grad: &mut [f32], dq_rot: &[f32], dk_rot: &[f32]) {
    for token in 0..TOKEN_COUNT {
        for head in 0..HEADS {
            for dim in 0..HEAD_DIM {
                let h = hidden_index(token, head, dim);
                let pair = hidden_index(token, head, dim ^ 1);
                for (offset, rot) in [(0, dq_rot), (EMBEDDING, dk_rot)] {
                    grad[qkv_index(token, head, dim, offset)] = rope_raw_grad(token, dim, rot[h], rot[pair]);
                }
            }
        }
    }
}

fn qkv_value(qkv: &[f32], token: usize, head: usize, dim: usize, offset: usize) -> f32 {
    qkv[qkv_index(token, head, dim, offset)]
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
    token as f32 * (-9.210_340_5 * ((dim & !1) as f32) / HEAD_DIM as f32).exp()
}
