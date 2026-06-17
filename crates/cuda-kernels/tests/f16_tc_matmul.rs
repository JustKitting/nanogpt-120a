use std::error::Error;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::f16_tc_matmul::{
    F16TcMatmulArgs, F16TcMatmulModule, F16TcMatmulScratch, f16_tc_matmul_elements,
};

mod common;

const BATCH: usize = 2;
const M: usize = 16;
const N: usize = 8;
const K: usize = 16;
const TOLERANCE: f32 = 1.0e-6;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn batched_f16_tc_matmul_matches_power_of_two_reference() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module = F16TcMatmulModule::from_module(ctx.load_module_from_file(&common::ptx_path())?)?;
    let a = DeviceBuffer::from_host(&stream, &vec![0.125_f32; BATCH * M * K])?;
    let b = DeviceBuffer::from_host(&stream, &vec![0.25_f32; BATCH * N * K])?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, BATCH * M * N)?;
    let mut scratch = ScratchBuffers::new(&stream)?;

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

    for value in out.to_host_vec(&stream)? {
        assert_close(value, 0.5);
    }
    Ok(())
}

struct ScratchBuffers {
    a_padded: DeviceBuffer<f32>,
    b_t_padded: DeviceBuffer<f32>,
    a_halves: DeviceBuffer<u16>,
    b_t_halves: DeviceBuffer<u16>,
}

impl ScratchBuffers {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        let a_rows = BATCH * M;
        let b_rows = BATCH * N;
        Ok(Self {
            a_padded: DeviceBuffer::zeroed(
                stream,
                f16_tc_matmul_elements(a_rows as u32, K as u32),
            )?,
            b_t_padded: DeviceBuffer::zeroed(
                stream,
                f16_tc_matmul_elements(b_rows as u32, K as u32),
            )?,
            a_halves: DeviceBuffer::zeroed(
                stream,
                f16_tc_matmul_elements(a_rows as u32, K as u32),
            )?,
            b_t_halves: DeviceBuffer::zeroed(
                stream,
                f16_tc_matmul_elements(b_rows as u32, K as u32),
            )?,
        })
    }

    fn args(&mut self) -> F16TcMatmulScratch<'_> {
        F16TcMatmulScratch {
            a_padded: &mut self.a_padded,
            b_t_padded: &mut self.b_t_padded,
            a_halves: &mut self.a_halves,
            b_t_halves: &mut self.b_t_halves,
        }
    }
}

fn assert_close(actual: f32, expected: f32) {
    let error = (actual - expected).abs();
    assert!(
        error <= TOLERANCE,
        "actual={actual:.8e} expected={expected:.8e} error={error:.8e}"
    );
}
