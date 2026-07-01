use std::fmt;

use gpt2_nvfp4::{HiddenState, GPT2_CONTEXT_LEN, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_QKV};

use crate::common::f16::tc_f16;

const HEAD_DIM: usize = GPT2_N_EMBD / GPT2_N_HEAD;
const ATTENTION_TOLERANCE: f32 = 1.0e-6;
const RESIDUAL_TOLERANCE: f32 = 1.0e-7;

pub fn assert_qkv_nonzero(qkv: &[f32]) {
    let nonzero = |start| qkv[start..start + GPT2_N_EMBD].iter().any(|value| value.abs() > 1.0e-7);
    assert!(nonzero(0) && nonzero(GPT2_N_EMBD) && nonzero(2 * GPT2_N_EMBD));
}

pub fn assert_attention_log_sum_exp(log_sum_exp: &[f32]) {
    assert!(log_sum_exp.iter().all(|value| value.is_finite()));
    assert!(log_sum_exp.iter().any(|value| value.abs() > 1.0e-7));
}

pub fn assert_attention_matches(qkv: &[f32], out: &[f32]) {
    for row in [0, 1, 2, 17, 128, GPT2_CONTEXT_LEN - 1] {
        for head in [0, GPT2_N_HEAD / 2, GPT2_N_HEAD - 1] {
            let scores = attention_scores(qkv, row, head);
            let score_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = scores
                .iter()
                .map(|score| (*score - score_max).exp())
                .sum::<f32>();

            for dim in [0, 1, HEAD_DIM / 2, HEAD_DIM - 1] {
                let mut expected = 0.0;
                for (key, score) in scores.iter().copied().enumerate() {
                    let weight = tc_f16((score - score_max).exp() / denom);
                    let value = tc_f16(qkv_value(qkv, key, head, dim, 2 * GPT2_N_EMBD));
                    expected += weight * value;
                }

                let col = head * HEAD_DIM + dim;
                assert_scaled_close(format_args!("row={row} head={head} dim={dim}"), out[row * GPT2_N_EMBD + col], expected, ATTENTION_TOLERANCE);
            }
        }
    }
}

fn attention_scores(qkv: &[f32], query: usize, head: usize) -> Vec<f32> {
    let mut scores = Vec::with_capacity(query + 1);
    for key in 0..=query {
        let mut dot = 0.0;
        for dim in 0..HEAD_DIM {
            let q = tc_f16(qkv_value(qkv, query, head, dim, 0));
            let k = tc_f16(qkv_value(qkv, key, head, dim, GPT2_N_EMBD));
            dot += q * k;
        }
        scores.push(dot / (HEAD_DIM as f32).sqrt());
    }
    scores
}

fn qkv_value(qkv: &[f32], token: usize, head: usize, dim: usize, offset: usize) -> f32 {
    qkv[token * GPT2_QKV + offset + head * HEAD_DIM + dim]
}

pub fn assert_output_amax(out: &[f32], output_amax: &[f32]) {
    for (row, actual) in output_amax.iter().copied().enumerate() {
        let row_base = row * GPT2_N_EMBD;
        let expected = out[row_base..row_base + GPT2_N_EMBD]
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        assert_scaled_close(format_args!("row={row}"), actual, expected, 1.0e-7);
    }
}

pub fn assert_c_proj_residual_add(
    residual_before: &[f32],
    attention_out: &[f32],
    residual_after: &[f32],
) {
    for index in 0..HiddenState::LEN {
        let expected = residual_before[index] + attention_out[index];
        assert_scaled_close(format_args!("index={index}"), residual_after[index], expected, RESIDUAL_TOLERANCE);
    }
}

fn assert_scaled_close(context: fmt::Arguments<'_>, actual: f32, expected: f32, relative_tolerance: f32) {
    let error = (actual - expected).abs();
    let tolerance = expected.abs().max(1.0) * relative_tolerance;
    assert!(
        error <= tolerance,
        "{context} actual={actual:.8e} expected={expected:.8e} error={error:.8e} tolerance={tolerance:.8e}"
    );
}
