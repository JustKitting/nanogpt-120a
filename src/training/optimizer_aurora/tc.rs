use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulArgs;

use super::super::optimizer_tc_scratch::TcMatmulScratch;
use super::AuroraModules;

#[allow(clippy::too_many_arguments)]
pub(super) fn tc_matmul(
    stream: &CudaStream,
    modules: AuroraModules<'_>,
    scratch: &mut TcMatmulScratch,
    a: &DeviceBuffer<f32>,
    b_t: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
    _base_seed: u32,
    _seed: u32,
) -> Result<(), DriverError> {
    modules.tc.batched_matmul(F16TcMatmulArgs {
        stream,
        a,
        b_t,
        out,
        scratch: scratch.scratch(),
        batch_count: 1,
        m,
        n,
        k,
    })
}
