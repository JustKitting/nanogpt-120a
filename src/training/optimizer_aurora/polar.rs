use cuda_core::DriverError;
use rust_kernels_cuda::transpose::TransposeF32Args;

use super::AuroraMatrixArgs;
use super::polar_express::polar_express_iteration;
use super::polar_source::{normalize_source_to_polar_x, transpose_source_to_polar_x};

pub(super) const POLAR_ITERATIONS: usize = 5;

#[derive(Clone, Copy)]
pub(super) enum PolarSource {
    Oriented,
    Scaled,
}

pub(super) fn polar(
    args: &mut AuroraMatrixArgs<'_, '_>,
    source: PolarSource,
    rows: u32,
    cols: u32,
) -> Result<(), DriverError> {
    if rows > cols {
        transpose_source_to_polar_x(args, source, rows, cols)?;
        normalize_polar_x(args, rows * cols)?;
        polar_wide(args, cols, rows)?;
        return args.modules.transpose.transpose_f32(TransposeF32Args {
            stream: args.stream,
            input: &args.scratch.polar_x,
            output: &mut args.scratch.u,
            rows: cols,
            cols: rows,
        });
    }

    normalize_source_to_polar_x(args, source, rows * cols)?;
    polar_wide(args, rows, cols)?;
    args.modules.optimizer.elementwise_linear_combination(
        args.stream,
        &args.scratch.polar_x,
        &args.scratch.polar_x,
        &mut args.scratch.u,
        1.0,
        0.0,
        rows * cols,
    )
}

fn polar_wide(
    args: &mut AuroraMatrixArgs<'_, '_>,
    rows: u32,
    cols: u32,
) -> Result<(), DriverError> {
    for iter in 0..POLAR_ITERATIONS {
        polar_express_iteration(args, rows, cols, iter)?;
    }
    Ok(())
}

fn normalize_polar_x(args: &mut AuroraMatrixArgs<'_, '_>, len: u32) -> Result<(), DriverError> {
    args.modules.optimizer.polar_normalize_in_place(
        args.stream,
        &mut args.scratch.polar_x,
        &mut args.scratch.polar_chunks,
        &mut args.scratch.polar_inv_norm,
        len,
    )
}
