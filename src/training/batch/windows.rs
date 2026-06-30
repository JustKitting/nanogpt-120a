use crate::AppResult;

pub(super) fn fill(
    windows: &[u16],
    batch_size: usize,
    seq_len: usize,
    tokens: &mut [u32],
    targets: &mut [u32],
) -> AppResult {
    let window_len = seq_len + 1;
    let needed = batch_size * window_len;
    if windows.len() < needed {
        return Err(format!(
            "token window has {} tokens, needs {}",
            windows.len(),
            needed
        )
        .into());
    }

    for batch in 0..batch_size {
        let window_base = batch * window_len;
        let out_base = batch * seq_len;
        for col in 0..seq_len {
            tokens[out_base + col] = windows[window_base + col] as u32;
            targets[out_base + col] = windows[window_base + col + 1] as u32;
        }
    }

    Ok(())
}
