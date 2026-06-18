use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::f16_tc_matmul::{F16TcMatmulAddArgs, F16TcSymmetricMatmulArgs};

use super::super::optimizer_tc_scratch::TcMatmulScratch;
use super::AuroraModules;

#[allow(clippy::too_many_arguments)]
pub(super) fn tc_matmul_add(
    stream: &CudaStream,
    modules: AuroraModules<'_>,
    scratch: &mut TcMatmulScratch,
    a: &DeviceBuffer<f32>,
    b_t: &DeviceBuffer<f32>,
    base: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
    base_scale: f32,
    matmul_scale: f32,
    _base_seed: u32,
    _seed: u32,
) -> Result<(), DriverError> {
    modules.tc.batched_matmul_add(F16TcMatmulAddArgs {
        stream,
        a,
        b_t,
        base,
        out,
        scratch: scratch.scratch(),
        batch_count: 1,
        m,
        n,
        k,
        base_scale,
        matmul_scale,
    })
}

pub(super) fn tc_self_matmul_symmetric(
    stream: &CudaStream,
    modules: AuroraModules<'_>,
    scratch: &mut TcMatmulScratch,
    x: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    rows: u32,
    cols: u32,
    _base_seed: u32,
    _seed: u32,
) -> Result<(), DriverError> {
    modules.tc.symmetric_matmul(F16TcSymmetricMatmulArgs {
        stream,
        x,
        out,
        scratch: scratch.scratch(),
        rows,
        cols,
    })
}
