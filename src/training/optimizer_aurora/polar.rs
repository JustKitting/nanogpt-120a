use cuda_core::DriverError;
use rust_kernels_cuda::transpose::TransposeF32Args;

use super::AuroraMatrixArgs;
use super::iteration::polar_iteration;
use super::polar_source::{copy_source_to_polar_x, transpose_source_to_polar_x};

pub(super) const POLAR_ITERATIONS: usize = 12;

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
        polar_wide(args, cols, rows)?;
        return args.modules.transpose.transpose_f32(TransposeF32Args {
            stream: args.stream,
            input: &args.scratch.polar_x,
            output: &mut args.scratch.u,
            rows: cols,
            cols: rows,
        });
    }

    copy_source_to_polar_x(args, source, rows * cols)?;
    polar_wide(args, rows, cols)?;
    args.modules.optimizer.matrix_combine(
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
    normalize(args, rows * cols)?;
    for iter in 0..POLAR_ITERATIONS {
        polar_iteration(args, rows, cols, iter)?;
    }
    Ok(())
}

fn normalize(args: &mut AuroraMatrixArgs<'_, '_>, len: u32) -> Result<(), DriverError> {
    args.modules.optimizer.matrix_frobenius_norm(
        args.stream,
        &args.scratch.polar_x,
        &mut args.scratch.norm,
        len,
    )?;
    args.modules.optimizer.matrix_scale_in_place(
        args.stream,
        &mut args.scratch.polar_x,
        &args.scratch.norm,
        len,
    )
}
