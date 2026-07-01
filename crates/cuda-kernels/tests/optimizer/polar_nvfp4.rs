use std::error::Error;

use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tc_matmul::Nvfp4TcMatmulModule;

use crate::common;

#[path = "polar_nvfp4/correction_reports.rs"]
mod correction_reports;
#[path = "polar_nvfp4/device.rs"]
mod device;
#[path = "polar_nvfp4/estimator.rs"]
mod estimator;
#[path = "polar_nvfp4/math.rs"]
mod math;
#[path = "polar_nvfp4/mode_cases.rs"]
mod mode_cases;
#[path = "polar_nvfp4/schedule/mod.rs"]
mod schedule;
#[path = "polar_nvfp4/schedule_reports.rs"]
mod schedule_reports;
#[path = "polar_nvfp4/scratch.rs"]
mod scratch;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ROWS: usize = 32;
const COLS: usize = 64;
const MAX_ITERATIONS: usize = 8;
const PRODUCTION_ITERATIONS: usize = 5;

fn with_polar<T>(run: impl FnOnce(&device::Nvfp4Polar<'_>) -> TestResult<T>) -> TestResult<T> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let f16 = F16TcMatmulModule::from_module(ptx.clone())?;
    let matmul = Nvfp4TcMatmulModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;
    let polar = device::Nvfp4Polar::new(&stream, &f16, &matmul, &quant);
    run(&polar)
}
