use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::f16_tc_matmul::{F16TcMatmulF32Args, F16TcMatmulModule};

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
    a: &DeviceBuffer<f32>,
    b_t: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DriverError> {
    tc_module.batched_matmul_f32_input(F16TcMatmulF32Args {
        stream,
        a,
        b_t,
        out,
        batch_count,
        m,
        n,
        k,
    })
}
