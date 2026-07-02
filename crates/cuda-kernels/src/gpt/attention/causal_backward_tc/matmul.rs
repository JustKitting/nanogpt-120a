#![expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]

use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::f16_tc_matmul::{
    F16TcMatmulF32ATransposedHalfRhsArgs, F16TcMatmulF32HalfRhsArgs, F16TcMatmulHalfArgs,
    F16TcMatmulModule,
};

pub(super) struct AttentionTcMatmulContext<'a> {
    pub stream: &'a CudaStream,
    pub tc_module: &'a F16TcMatmulModule,
    pub batch_head: u32,
    pub seq_len: u32,
    pub head_dim: u32,
}

pub(super) fn run_tc_matmul(
    stream: &CudaStream,
    tc_module: &F16TcMatmulModule,
    a: &DeviceBuffer<u16>,
    b_t: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<f32>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DriverError> {
    tc_module.batched_matmul_half_input_lower(F16TcMatmulHalfArgs {
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

pub(super) fn run_tc_matmul_rhs(
    stream: &CudaStream,
    tc_module: &F16TcMatmulModule,
    a: &DeviceBuffer<f32>,
    rhs: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<f32>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DriverError> {
    tc_module.batched_matmul_f32_half_rhs_lower_a(F16TcMatmulF32HalfRhsArgs {
        stream,
        a,
        rhs,
        out,
        batch_count,
        m,
        n,
        k,
    })
}

pub(super) fn run_tc_matmul_a_transposed_rhs(
    stream: &CudaStream,
    tc_module: &F16TcMatmulModule,
    a: &DeviceBuffer<f32>,
    rhs: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<f32>,
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), DriverError> {
    tc_module.batched_matmul_f32_a_transposed_half_rhs_lower_a(
        F16TcMatmulF32ATransposedHalfRhsArgs {
            stream,
            a,
            rhs,
            out,
            batch_count,
            m,
            n,
            k,
        },
    )
}
