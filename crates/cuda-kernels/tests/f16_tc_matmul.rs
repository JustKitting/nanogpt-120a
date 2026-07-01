use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::f16_tc_matmul::{F16TcMatmulArgs, F16TcMatmulModule};

mod common;

use common::f16_tc::F16TcScratchBuffers;

const BATCH: usize = 2;
const M: usize = 16;
const N: usize = 8;
const K: usize = 16;
const TOLERANCE: f32 = 1.0e-6;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn batched_f16_tc_matmul_matches_power_of_two_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(F16TcMatmulModule::from_module)?;
    let a = DeviceBuffer::from_host(&stream, &vec![0.125_f32; BATCH * M * K])?;
    let b = DeviceBuffer::from_host(&stream, &vec![0.25_f32; BATCH * N * K])?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, BATCH * M * N)?;
    let mut scratch = F16TcScratchBuffers::new(&stream, (BATCH * M, BATCH * N, K))?;

    module.batched_matmul(F16TcMatmulArgs {
        stream: &stream,
        a: &a,
        b_t: &b,
        out: &mut out,
        scratch: scratch.args(),
        batch_count: BATCH as u32,
        m: M as u32,
        n: N as u32,
        k: K as u32,
    })?;

    common::assert_all_close(&out.to_host_vec(&stream)?, 0.5, TOLERANCE);
    Ok(())
}
