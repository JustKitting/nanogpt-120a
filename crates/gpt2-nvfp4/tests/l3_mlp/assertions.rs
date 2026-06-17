use gpt2_nvfp4::{GPT2_CONTEXT_LEN, GPT2_MLP, GPT2_N_EMBD};

pub fn assert_relu2_samples(activation: &[f32]) {
    for row in [0, 1, 17, GPT2_CONTEXT_LEN - 1] {
        assert_positive_relu2(activation, row, 0);
        assert_positive_relu2(activation, row, 37);
        assert_positive_relu2(activation, row, GPT2_N_EMBD + 11);
        assert_zero_relu2(activation, row, GPT2_N_EMBD / 2);
        assert_zero_relu2(activation, row, GPT2_N_EMBD + GPT2_N_EMBD / 2 + 5);
    }
}

pub fn assert_down_projection_residual_add(residual_before: &[f32], residual_after: &[f32]) {
    for row in [0, 1, 17, GPT2_CONTEXT_LEN - 1] {
        assert_residual_delta(residual_before, residual_after, row, 0, 0.25);
        assert_residual_delta(residual_before, residual_after, row, 37, 0.25);
        assert_residual_delta(residual_before, residual_after, row, GPT2_N_EMBD / 2, 0.0);
        assert_residual_delta(residual_before, residual_after, row, GPT2_N_EMBD - 1, 0.0);
    }
}

fn assert_positive_relu2(activation: &[f32], row: usize, col: usize) {
    let actual = activation[row * GPT2_MLP + col];
    let expected = 0.25_f32;
    let error = (actual - expected).abs();
    assert!(
        error <= 5.0e-2,
        "row={row} col={col} actual={actual:.8e} expected={expected:.8e} error={error:.8e}"
    );
}

fn assert_zero_relu2(activation: &[f32], row: usize, col: usize) {
    let actual = activation[row * GPT2_MLP + col];
    assert!(
        actual.abs() <= 1.0e-6,
        "row={row} col={col} actual={actual:.8e}"
    );
}

fn assert_residual_delta(
    residual_before: &[f32],
    residual_after: &[f32],
    row: usize,
    col: usize,
    expected_delta: f32,
) {
    let index = row * GPT2_N_EMBD + col;
    let actual_delta = residual_after[index] - residual_before[index];
    let error = (actual_delta - expected_delta).abs();
    assert!(
        error <= 5.0e-2,
        "row={row} col={col} actual_delta={actual_delta:.8e} expected_delta={expected_delta:.8e} error={error:.8e}"
    );
}
