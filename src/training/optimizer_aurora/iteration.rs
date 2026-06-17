use cuda_core::DriverError;
use rust_kernels_cuda::transpose::TransposeF32Args;

use super::AuroraMatrixArgs;
use super::polar::POLAR_ITERATIONS;
use super::tc::tc_matmul;

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
    tc_matmul(
        args.stream,
        args.modules,
        &mut args.scratch.tc,
        &args.scratch.polar_x,
        &args.scratch.polar_x,
        &mut args.scratch.a,
        rows,
        rows,
        cols,
        args.seed,
        0x1000 + iter as u32,
    )?;
    args.modules.transpose.transpose_f32(TransposeF32Args {
        stream: args.stream,
        input: &args.scratch.a,
        output: &mut args.scratch.a_t,
        rows,
        cols: rows,
    })
}

fn build_b(args: &mut AuroraMatrixArgs<'_, '_>, rows: u32, iter: usize) -> Result<(), DriverError> {
    tc_matmul(
        args.stream,
        args.modules,
        &mut args.scratch.tc,
        &args.scratch.a,
        &args.scratch.a_t,
        &mut args.scratch.aa,
        rows,
        rows,
        rows,
        args.seed,
        0x2000 + iter as u32,
    )?;
    args.modules.optimizer.matrix_combine(
        args.stream,
        &args.scratch.a,
        &args.scratch.aa,
        &mut args.scratch.b,
        -1.5,
        0.5,
        rows * rows,
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
    tc_matmul(
        args.stream,
        args.modules,
        &mut args.scratch.tc,
        &args.scratch.b,
        &args.scratch.polar_xt,
        &mut args.scratch.bx,
        rows,
        cols,
        rows,
        args.seed,
        0x3000 + iter as u32,
    )?;
    args.modules.optimizer.matrix_combine(
        args.stream,
        &args.scratch.polar_x,
        &args.scratch.bx,
        &mut args.scratch.polar_next,
        2.0,
        1.0,
        rows * cols,
    )
}
