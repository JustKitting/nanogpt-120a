use cuda_core::DriverError;

use super::AuroraMatrixArgs;
use super::tc::{tc_matmul_add, tc_matmul_add_rhs_transposed_in_place, tc_self_matmul_symmetric};

pub(super) fn polar_iteration(
    args: &mut AuroraMatrixArgs<'_, '_>,
    rows: u32,
    cols: u32,
    iter: usize,
) -> Result<(), DriverError> {
    build_a(args, rows, cols, iter)?;
    build_b(args, rows, iter)?;
    apply_b(args, rows, cols)
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

fn apply_b(args: &mut AuroraMatrixArgs<'_, '_>, rows: u32, cols: u32) -> Result<(), DriverError> {
    tc_matmul_add_rhs_transposed_in_place(
        args.stream,
        args.modules,
        &mut args.scratch.tc,
        &args.scratch.b,
        &mut args.scratch.polar_x,
        rows,
        cols,
        rows,
        2.0,
        1.0,
    )
}
