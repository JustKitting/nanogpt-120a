use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::f16_tc_matmul::{
    F16TcMatmulAddArgs, F16TcMatmulAddRhsTransposeBaseArgs, F16TcMatmulModule,
};

mod common;

use common::f16_tc::F16TcScratchBuffers;

const BATCH: usize = 1;
const M: usize = 64;
const N: usize = 64;
const K: usize = 32;
const TOLERANCE: f32 = 1.0e-6;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn cta_tiled_f16_tc_matmul_add_matches_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(F16TcMatmulModule::from_module)?;
    let a = DeviceBuffer::from_host(&stream, &vec![0.125_f32; BATCH * M * K])?;
    let b = DeviceBuffer::from_host(&stream, &vec![0.25_f32; BATCH * N * K])?;
    let base = DeviceBuffer::from_host(&stream, &vec![0.5_f32; BATCH * M * N])?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, BATCH * M * N)?;
    let mut scratch = F16TcScratchBuffers::new(&stream, (BATCH * M, BATCH * N, K))?;

    module.batched_matmul_add(F16TcMatmulAddArgs {
        stream: &stream,
        a: &a,
        b_t: &b,
        base: &base,
        out: &mut out,
        scratch: scratch.args(),
        batch_count: BATCH as u32,
        m: M as u32,
        n: N as u32,
        k: K as u32,
        base_scale: 2.0,
        matmul_scale: 3.0,
    })?;

    common::assert_all_close(&out.to_host_vec(&stream)?, 4.0, TOLERANCE);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn cta_tiled_f16_tc_rhs_transposed_base_matches_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(F16TcMatmulModule::from_module)?;
    let a = DeviceBuffer::from_host(&stream, &vec![0.125_f32; BATCH * M * K])?;
    let rhs = DeviceBuffer::from_host(&stream, &vec![0.25_f32; BATCH * K * N])?;
    let base = DeviceBuffer::from_host(&stream, &vec![0.5_f32; BATCH * M * N])?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, BATCH * M * N)?;

    module.batched_matmul_add_rhs_transposed_base(F16TcMatmulAddRhsTransposeBaseArgs {
        stream: &stream,
        a: &a,
        rhs: &rhs,
        base: &base,
        out: &mut out,
        batch_count: BATCH as u32,
        m: M as u32,
        n: N as u32,
        k: K as u32,
        base_scale: 2.0,
        matmul_scale: 3.0,
    })?;

    common::assert_all_close(&out.to_host_vec(&stream)?, 4.0, TOLERANCE);
    Ok(())
}
