use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tc_matmul::{Nvfp4TcMatmulArgs, Nvfp4TcMatmulModule};

mod common;
#[path = "nvfp4_tc_matmul/decode.rs"]
mod decode;
#[path = "nvfp4_tc_matmul/scratch.rs"]
mod scratch;

use decode::decoded_dot;
use scratch::{ScratchBuffers, K, M, N};

const TOLERANCE: f32 = 1.0e-5;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn fp32_ms_eden_tc_matmul_matches_decoded_operands() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = Nvfp4TcMatmulModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx.clone())?;
    let decode = Nvfp4DecodeModule::from_module(ptx)?;

    let a_dev = DeviceBuffer::from_host(&stream, &vec![1.0_f32; M * K])?;
    let b_t_dev = DeviceBuffer::from_host(&stream, &vec![0.5_f32; N * K])?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, M * N)?;
    let mut scratch = ScratchBuffers::new(&stream, (M, N, K), (1.0, 1.0))?;

    module.matmul_ms_eden(Nvfp4TcMatmulArgs {
        stream: &stream,
        quant_module: &quant,
        a: &a_dev,
        b_t: &b_t_dev,
        out: &mut out,
        scratch: scratch.args(),
        m: M as u32,
        n: N as u32,
        k: K as u32,
        sign_seed: 0x1234_5678,
        scale_seed: 0x9abc_def0,
    })?;

    let expected = decoded_dot(&decode, &stream, &scratch)?;
    common::assert_close(out.to_host_vec(&stream)?[0], expected, TOLERANCE);
    Ok(())
}
