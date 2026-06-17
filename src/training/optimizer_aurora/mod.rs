mod apply;
mod balance;
mod finalize;
mod iteration;
mod orient;
mod polar;
mod polar_source;
mod tc;

use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tc_matmul::Nvfp4TcMatmulModule;
use rust_kernels_cuda::optimizer::OptimizerModule;
use rust_kernels_cuda::transpose::TransposeModule;

use crate::upload::UploadedNvfp4;

use super::optimizer::OptimizerScratch;
use super::optimizer_state::AuroraState;
use super::optimizer_tc_scratch::AuroraScratchBuffers;

use apply::apply_update;
use balance::aurora_oriented;
use finalize::finalize_update;
use orient::orient_update;

pub(super) const PP_ITERATIONS: usize = 2;
pub(super) const EPS: f32 = 1.0e-7;

const MU: f32 = 0.95;
pub(super) const AURORA_LR: f32 = 3.75e-3;
pub(super) const AURORA_WEIGHT_DECAY: f32 = 0.025;

#[derive(Clone, Copy)]
pub(super) struct AuroraModules<'a> {
    pub(super) optimizer: &'a OptimizerModule,
    pub(super) quant: &'a Nvfp4QuantModule,
    pub(super) tc: &'a Nvfp4TcMatmulModule,
    pub(super) transpose: &'a TransposeModule,
}

pub(super) struct AuroraMatrixArgs<'a, 'scratch> {
    pub(super) stream: &'a CudaStream,
    pub(super) modules: AuroraModules<'a>,
    pub(super) tensor: &'a mut UploadedNvfp4,
    pub(super) grad: &'a DeviceBuffer<f32>,
    pub(super) state: &'a mut AuroraState,
    pub(super) scratch: &'scratch mut AuroraScratchBuffers,
    pub(super) optimizer_scratch: &'a mut OptimizerScratch,
    pub(super) rows: u32,
    pub(super) cols: u32,
    pub(super) seed: u32,
}

pub(super) fn apply_aurora_matrix(args: AuroraMatrixArgs<'_, '_>) -> Result<(), DriverError> {
    let mut args = args;
    let len = args.rows * args.cols;
    args.modules.optimizer.aurora_momentum(
        args.stream,
        args.grad,
        &mut args.state.momentum,
        &mut args.scratch.update,
        MU,
        len,
    )?;

    let (oriented_rows, oriented_cols, transposed) = orient_update(&mut args)?;
    aurora_oriented(&mut args, oriented_rows, oriented_cols)?;
    finalize_update(&mut args, len, transposed)?;
    apply_update(args, len)
}
