use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::runtime::Runtime;
use crate::upload::UploadedNvfp4;

use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_aurora::{AuroraMatrixArgs, AuroraModules, apply_aurora_matrix};
use super::super::optimizer_state::LinearState;
use super::super::optimizer_tc_scratch::AuroraScratchBuffers;
use super::adam::update_adam_tensor;

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum MatrixOptimizer {
    Adam,
    Aurora,
}

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
    step: u32,
    seed: u32,
) -> Result<MatrixOptimizer, DriverError> {
    if !use_aurora() {
        update_adam_tensor(
            stream,
            &runtime.optimizer,
            tensor,
            grad,
            scratch,
            &mut state.weight_adam,
            step,
        )?;
        return Ok(MatrixOptimizer::Adam);
    }

    apply_aurora_matrix(AuroraMatrixArgs {
        stream,
        modules: AuroraModules {
            optimizer: &runtime.optimizer,
            quant: &runtime.quant,
            tc: &runtime.tc_matmul,
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
    })?;
    Ok(MatrixOptimizer::Aurora)
}

fn use_aurora() -> bool {
    matches!(
        std::env::var("TRAIN_MATRIX_OPTIMIZER"),
        Ok(value) if value.eq_ignore_ascii_case("aurora")
    )
}
