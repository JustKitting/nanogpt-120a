use cuda_core::DriverError;

use super::AuroraMatrixArgs;
use super::polar_express_coefficients::{PolarExpressCoefficients, polar_express_coefficients};
use super::tc::{tc_matmul_add, tc_matmul_add_rhs_transposed, tc_self_matmul_symmetric};

pub(super) fn polar_express_iteration(
    args: &mut AuroraMatrixArgs<'_, '_>,
    rows: u32,
    cols: u32,
    iter: usize,
) -> Result<(), DriverError> {
    let coefficients = polar_express_coefficients(iter);
    build_a(args, rows, cols, iter)?;
    build_b(args, rows, iter, coefficients)?;
    apply_b(args, rows, cols, coefficients)
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

fn build_b(
    args: &mut AuroraMatrixArgs<'_, '_>,
    rows: u32,
    iter: usize,
    coefficients: PolarExpressCoefficients,
) -> Result<(), DriverError> {
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
        coefficients.b,
        coefficients.c,
        args.seed,
        0x2000 + iter as u32,
    )
}

fn apply_b(
    args: &mut AuroraMatrixArgs<'_, '_>,
    rows: u32,
    cols: u32,
    coefficients: PolarExpressCoefficients,
) -> Result<(), DriverError> {
    tc_matmul_add_rhs_transposed(
        args.stream,
        args.modules,
        &mut args.scratch.tc,
        &args.scratch.b,
        &args.scratch.polar_x,
        &mut args.scratch.oriented,
        rows,
        cols,
        rows,
        coefficients.a,
        1.0,
    )?;
    std::mem::swap(&mut args.scratch.polar_x, &mut args.scratch.oriented);
    Ok(())
}
