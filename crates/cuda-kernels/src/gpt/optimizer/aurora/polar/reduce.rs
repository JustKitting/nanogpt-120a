use super::super::super::POLAR_SUM_VALUES_PER_BLOCK;
use super::super::super::threads::MATRIX_THREADS_PER_BLOCK;

#[inline(always)]
pub(super) fn input_chunk_sum(x: &[f32], base: u32, tid: u32, len: u32) -> f32 {
    let stride = MATRIX_THREADS_PER_BLOCK;
    let i0 = base + tid;
    let i1 = i0 + stride;
    let i2 = i1 + stride;
    let i3 = i2 + stride;

    if base + POLAR_SUM_VALUES_PER_BLOCK as u32 <= len {
        return square(x[i0 as usize])
            + square(x[i1 as usize])
            + square(x[i2 as usize])
            + square(x[i3 as usize]);
    }

    checked_square(x, i0, len)
        + checked_square(x, i1, len)
        + checked_square(x, i2, len)
        + checked_square(x, i3, len)
}

#[inline(always)]
pub(super) fn chunk_sum(chunks: &[f32], tid: u32, chunk_count: u32) -> f32 {
    let mut local = 0.0;
    let mut chunk = tid;

    while chunk < chunk_count {
        local += chunks[chunk as usize];
        chunk += MATRIX_THREADS_PER_BLOCK;
    }

    local
}

#[inline(always)]
fn checked_square(x: &[f32], index: u32, len: u32) -> f32 {
    if index < len {
        square(x[index as usize])
    } else {
        0.0
    }
}

#[inline(always)]
fn square(x: f32) -> f32 {
    x * x
}
