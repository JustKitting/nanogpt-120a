use cuda_core::DriverError;
use rust_kernels_cuda::transpose::TransposeF32Args;

use super::AuroraMatrixArgs;
use super::polar::POLAR_ITERATIONS;
use super::tc::{tc_matmul_add, tc_self_matmul_symmetric};

pub(super) fn polar_iteration(
    args: &mut AuroraMatrixArgs<'_, '_>,
    rows: u32,
    cols: u32,
    iter: usize,
) -> Result<(), DriverError> {
    build_a(args, rows, cols, iter)?;
    build_b(args, rows, iter)?;
    apply_b(args, rows, cols, iter)?;
    if iter + 1 < POLAR_ITERATIONS {
        args.modules.optimizer.matrix_combine(
            args.stream,
            &args.scratch.polar_next,
            &args.scratch.polar_next,
            &mut args.scratch.polar_x,
            1.0,
            0.0,
            rows * cols,
        )?;
    }
    Ok(())
}

fn build_a(
    args: &mut AuroraMatrixArgs<'_, '_>,
    rows: u32,
    cols: u32,
    iter: usize,
) -> Result<(), DriverError> {
    tc_self_matmul_symmetric(
        args.stream,
        args.modules,
        &mut args.scratch.tc,
        &args.scratch.polar_x,
        &mut args.scratch.a,
        rows,
        cols,
        args.seed,
        0x1000 + iter as u32,
    )
}

fn build_b(args: &mut AuroraMatrixArgs<'_, '_>, rows: u32, iter: usize) -> Result<(), DriverError> {
    tc_matmul_add(
        args.stream,
        args.modules,
        &mut args.scratch.tc,
        &args.scratch.a,
        &args.scratch.a,
        &args.scratch.a,
        &mut args.scratch.b,
        rows,
        rows,
        rows,
        -1.5,
        0.5,
        args.seed,
        0x2000 + iter as u32,
    )
}

fn apply_b(
    args: &mut AuroraMatrixArgs<'_, '_>,
    rows: u32,
    cols: u32,
    iter: usize,
) -> Result<(), DriverError> {
    args.modules.transpose.transpose_f32(TransposeF32Args {
        stream: args.stream,
        input: &args.scratch.polar_x,
        output: &mut args.scratch.polar_xt,
        rows,
        cols,
    })?;
    tc_matmul_add(
        args.stream,
        args.modules,
        &mut args.scratch.tc,
        &args.scratch.b,
        &args.scratch.polar_xt,
        &args.scratch.polar_x,
        &mut args.scratch.polar_next,
        rows,
        cols,
        rows,
        2.0,
        1.0,
        args.seed,
        0x3000 + iter as u32,
    )
}
