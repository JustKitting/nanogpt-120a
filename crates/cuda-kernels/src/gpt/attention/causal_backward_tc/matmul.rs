use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::f16_tc_matmul::{F16TcMatmulArgs, F16TcMatmulModule, F16TcMatmulScratch};

pub(super) struct AttentionTcMatmulContext<'a> {
    pub stream: &'a CudaStream,
    pub tc_module: &'a F16TcMatmulModule,
    pub batch_head: u32,
    pub seq_len: u32,
    pub head_dim: u32,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_tc_matmul(
    stream: &CudaStream,
    tc_module: &F16TcMatmulModule,
    scratch: &mut F16TcMatmulScratch<'_>,
    a: &DeviceBuffer<f32>,
    b_t: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DriverError> {
    tc_module.batched_matmul(F16TcMatmulArgs {
        stream,
        a,
        b_t,
        out,
        scratch: scratch.reborrow(),
        batch_count,
        m,
        n,
        k,
    })
}
