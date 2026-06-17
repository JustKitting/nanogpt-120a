use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::runtime::Runtime;
use crate::upload::UploadedNvfp4;

use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_aurora::{AuroraMatrixArgs, AuroraModules, apply_aurora_matrix};
use super::super::optimizer_state::LinearState;
use super::super::optimizer_tc_scratch::AuroraScratchBuffers;

pub(super) fn update_matrix_tensor(
    stream: &CudaStream,
    runtime: &Runtime,
    tensor: &mut UploadedNvfp4,
    grad: &DeviceBuffer<f32>,
    scratch: &mut OptimizerScratch,
    state: &mut LinearState,
    aurora: &mut AuroraScratchBuffers,
    rows: u32,
    cols: u32,
    seed: u32,
    step: u32,
) -> Result<(), DriverError> {
    apply_aurora_matrix(AuroraMatrixArgs {
        stream,
        modules: AuroraModules {
            optimizer: &runtime.optimizer,
            tc: &runtime.f16_tc_matmul,
            transpose: &runtime.transpose,
        },
        tensor,
        grad,
        state: &mut state.weight_aurora,
        scratch: aurora,
        optimizer_scratch: scratch,
        rows,
        cols,
        seed,
        step,
    })
}
