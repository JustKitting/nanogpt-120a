use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4_tc_matmul::Nvfp4TcMatmulArgs;

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
    base_seed: u32,
    seed: u32,
) -> Result<(), DriverError> {
    modules.tc.matmul_ms_eden(Nvfp4TcMatmulArgs {
        stream,
        quant_module: modules.quant,
        a,
        b_t,
        out,
        scratch: scratch.scratch(),
        m,
        n,
        k,
        sign_seed: base_seed ^ seed,
        scale_seed: base_seed.rotate_left(13) ^ seed,
    })
}
