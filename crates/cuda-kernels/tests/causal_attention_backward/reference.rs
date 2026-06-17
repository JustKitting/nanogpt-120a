use super::reference_math::{backward, logsumexp, scores};
use super::shape::*;

pub struct Case {
    pub qkv: Vec<f32>,
    pub d_out: Vec<f32>,
    pub out: Vec<f32>,
    pub lse: Vec<f32>,
    pub expected: Vec<f32>,
}

pub fn case() -> Case {
    let qkv = qkv_values();
    let d_out = hidden_values(0.011);
    let (out, lse) = forward(&qkv);
    let expected = backward(&qkv, &out, &d_out, &lse);
    Case {
        qkv,
        d_out,
        out,
        lse,
        expected,
    }
}

fn qkv_values() -> Vec<f32> {
    (0..TOKEN_COUNT * QKV_DIM)
        .map(|i| {
            let row = i / QKV_DIM;
            let col = i % QKV_DIM;
            (row as f32 * 0.07 + col as f32 * 0.013).sin() * 0.25
        })
        .collect()
}

fn hidden_values(scale: f32) -> Vec<f32> {
    (0..TOKEN_COUNT * EMBEDDING)
        .map(|i| {
            let row = i / EMBEDDING;
            let col = i % EMBEDDING;
            ((row * 17 + col * 5) as f32 * scale).cos() * 0.125
        })
        .collect()
}

fn forward(qkv: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0_f32; TOKEN_COUNT * EMBEDDING];
    let mut lse = vec![0.0_f32; TOKEN_COUNT * HEADS];
    for query in 0..TOKEN_COUNT {
        for head in 0..HEADS {
            let scores = scores(qkv, query, head);
            let row_lse = logsumexp(&scores);
            lse[lse_index(query, head)] = row_lse;
            for dim in 0..HEAD_DIM {
                let mut value = 0.0;
                for key in 0..=query {
                    let p = (scores[key] - row_lse).exp();
                    value += p * qkv[qkv_index(key, head, dim, 2 * EMBEDDING)];
                }
                out[hidden_index(query, head, dim)] = value;
            }
        }
    }
    (out, lse)
}
