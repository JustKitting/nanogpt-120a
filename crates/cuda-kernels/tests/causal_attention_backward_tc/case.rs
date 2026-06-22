use super::shape;

pub struct Case {
    pub qkv: Vec<f32>,
    pub d_out: Vec<f32>,
    pub out: Vec<f32>,
    pub log_sum_exp: Vec<f32>,
}

pub fn simple_case() -> Case {
    let mut qkv = vec![0.0_f32; shape::TOKEN_COUNT * shape::QKV_DIM];
    let mut d_out = vec![0.0_f32; shape::TOKEN_COUNT * shape::EMBEDDING];
    fill_qkv(&mut qkv);
    fill_d_out(&mut d_out);
    let out = causal_uniform_out(&qkv);
    let log_sum_exp = causal_zero_score_lse();
    Case {
        qkv,
        d_out,
        out,
        log_sum_exp,
    }
}

fn fill_qkv(qkv: &mut [f32]) {
    for token in 0..shape::TOKEN_COUNT {
        for head in 0..shape::HEADS {
            for dim in 0..shape::HEAD_DIM {
                let value = 0.125 + token as f32 * 0.03125 + dim as f32 * 0.015625;
                qkv[shape::qkv_index(token, head, dim, 0)] = if head == 0 { value } else { 0.0 };
                qkv[shape::qkv_index(token, head, dim, shape::EMBEDDING)] =
                    if head == 1 { value } else { 0.0 };
                qkv[shape::qkv_index(token, head, dim, 2 * shape::EMBEDDING)] = value * 0.5;
            }
        }
    }
}

fn fill_d_out(d_out: &mut [f32]) {
    for token in 0..shape::TOKEN_COUNT {
        for head in 0..shape::HEADS {
            for dim in 0..shape::HEAD_DIM {
                d_out[shape::hidden_index(token, head, dim)] =
                    0.0625 + token as f32 * 0.015625 + dim as f32 * 0.0078125;
            }
        }
    }
}

fn causal_uniform_out(qkv: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; shape::TOKEN_COUNT * shape::EMBEDDING];
    for query in 0..shape::TOKEN_COUNT {
        for head in 0..shape::HEADS {
            for dim in 0..shape::HEAD_DIM {
                out[shape::hidden_index(query, head, dim)] = v_prefix_avg(qkv, query, head, dim);
            }
        }
    }
    out
}

fn v_prefix_avg(qkv: &[f32], query: usize, head: usize, dim: usize) -> f32 {
    let mut value = 0.0;
    for key in 0..=query {
        value += qkv[shape::qkv_index(key, head, dim, 2 * shape::EMBEDDING)];
    }
    value / (query + 1) as f32
}

fn causal_zero_score_lse() -> Vec<f32> {
    let mut values = vec![0.0_f32; shape::TOKEN_COUNT * shape::HEADS];
    for head in 0..shape::HEADS {
        for token in 0..shape::TOKEN_COUNT {
            values[shape::log_sum_exp_index(token, head)] = ((token + 1) as f32).ln();
        }
    }
    values
}
