use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::args::{F16TcMatmulScratch, f16_tc_matmul_padded_k};
use super::kernels::LoadedModule;
use super::launch_ops::{convert, pad_rows};

pub(super) fn prepare_halves<'scratch>(
    module: &LoadedModule,
    stream: &CudaStream,
    a: &DeviceBuffer<f32>,
    b_t: &DeviceBuffer<f32>,
    scratch: F16TcMatmulScratch<'scratch>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(F16TcMatmulScratch<'scratch>, u32), DriverError> {
    let padded_k = f16_tc_matmul_padded_k(k);
    let a_rows = batch_count * m;
    let b_rows = batch_count * n;
    assert!(a.len() >= a_rows as usize * k as usize);
    assert!(b_t.len() >= b_rows as usize * k as usize);
    assert!(scratch.a_halves.len() >= a_rows as usize * padded_k as usize);
    assert!(scratch.b_t_halves.len() >= b_rows as usize * padded_k as usize);

    if padded_k == k {
        convert(module, stream, a, scratch.a_halves, a_rows * k)?;
        convert(module, stream, b_t, scratch.b_t_halves, b_rows * k)?;
        return Ok((scratch, k));
    }

    assert!(scratch.a_padded.len() >= a_rows as usize * padded_k as usize);
    assert!(scratch.b_t_padded.len() >= b_rows as usize * padded_k as usize);
    pad_rows(module, stream, a, scratch.a_padded, a_rows, k, padded_k)?;
    pad_rows(module, stream, b_t, scratch.b_t_padded, b_rows, k, padded_k)?;
    convert(
        module,
        stream,
        scratch.a_padded,
        scratch.a_halves,
        a_rows * padded_k,
    )?;
    convert(
        module,
        stream,
        scratch.b_t_padded,
        scratch.b_t_halves,
        b_rows * padded_k,
    )?;
    Ok((scratch, padded_k))
}

pub(super) fn prepare_self_halves<'scratch>(
    module: &LoadedModule,
    stream: &CudaStream,
    x: &DeviceBuffer<f32>,
    scratch: F16TcMatmulScratch<'scratch>,
    rows: u32,
    cols: u32,
) -> Result<(F16TcMatmulScratch<'scratch>, u32), DriverError> {
    let padded_cols = f16_tc_matmul_padded_k(cols);
    assert!(x.len() >= rows as usize * cols as usize);
    assert!(scratch.a_halves.len() >= rows as usize * padded_cols as usize);

    if padded_cols == cols {
        convert(module, stream, x, scratch.a_halves, rows * cols)?;
        return Ok((scratch, cols));
    }

    assert!(scratch.a_padded.len() >= rows as usize * padded_cols as usize);
    pad_rows(module, stream, x, scratch.a_padded, rows, cols, padded_cols)?;
    convert(
        module,
        stream,
        scratch.a_padded,
        scratch.a_halves,
        rows * padded_cols,
    )?;
    Ok((scratch, padded_cols))
}
