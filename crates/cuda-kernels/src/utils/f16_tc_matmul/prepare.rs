use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::args::{F16TcMatmulScratch, f16_tc_matmul_padded_k};
use super::cta_tile::CtaMatmulDims;
use super::kernels::LoadedModule;
use super::launch_ops::{convert, pad_rows};

pub(super) fn prepare_halves<'scratch>(
    module: &LoadedModule,
    stream: &CudaStream,
    a: &DeviceBuffer<f32>,
    b_t: &DeviceBuffer<f32>,
    scratch: F16TcMatmulScratch<'scratch>,
    dims: CtaMatmulDims,
) -> Result<(F16TcMatmulScratch<'scratch>, u32), DriverError> {
    let padded_k = f16_tc_matmul_padded_k(dims.k);
    let a_rows = dims.batch_count * dims.m;
    let b_rows = dims.batch_count * dims.n;
    assert!(a.len() >= a_rows as usize * dims.k as usize);
    assert!(b_t.len() >= b_rows as usize * dims.k as usize);
    assert!(scratch.a_halves.len() >= a_rows as usize * padded_k as usize);
    assert!(scratch.b_t_halves.len() >= b_rows as usize * padded_k as usize);

    if padded_k == dims.k {
        convert(module, stream, a, scratch.a_halves, a_rows * dims.k)?;
        convert(module, stream, b_t, scratch.b_t_halves, b_rows * dims.k)?;
        return Ok((scratch, dims.k));
    }

    assert!(scratch.a_padded.len() >= a_rows as usize * padded_k as usize);
    assert!(scratch.b_t_padded.len() >= b_rows as usize * padded_k as usize);
    pad_rows(
        module,
        stream,
        a,
        scratch.a_padded,
        a_rows,
        dims.k,
        padded_k,
    )?;
    pad_rows(
        module,
        stream,
        b_t,
        scratch.b_t_padded,
        b_rows,
        dims.k,
        padded_k,
    )?;
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
