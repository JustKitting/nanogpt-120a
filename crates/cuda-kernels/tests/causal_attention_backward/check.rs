use super::shape::{EMBEDDING, HEAD_DIM, HEADS, QKV_DIM, TOKEN_COUNT, qkv_index};

const TOLERANCE: f32 = 1.0e-7;

pub fn assert_grad_close(actual: &[f32], expected: &[f32]) {
    assert_section_close("dQ", actual, expected, 0);
    assert_section_close("dK", actual, expected, EMBEDDING);
    assert_section_close("dV", actual, expected, 2 * EMBEDDING);
}

fn assert_section_close(name: &str, actual: &[f32], expected: &[f32], offset: usize) {
    let mut max_error = 0.0_f32;
    let mut max_index = 0_usize;
    for token in 0..TOKEN_COUNT {
        for head in 0..HEADS {
            for dim in 0..HEAD_DIM {
                let index = qkv_index(token, head, dim, offset);
                let error = (actual[index] - expected[index]).abs();
                if error > max_error {
                    max_error = error;
                    max_index = index;
                }
            }
        }
    }
    assert!(
        max_error <= TOLERANCE,
        "{name} max_error={max_error:.8e} index={max_index} actual={:.8e} expected={:.8e} qkv_dim={QKV_DIM}",
        actual[max_index],
        expected[max_index],
    );
}
