use cuda_core::DriverError;

use super::polar::{PolarSource, polar};
use super::{AuroraMatrixArgs, EPS, PP_ITERATIONS};

pub(super) fn aurora_oriented(
    args: &mut AuroraMatrixArgs<'_, '_>,
    rows: u32,
    cols: u32,
) -> Result<(), DriverError> {
    if rows == cols {
        return polar(args, PolarSource::Oriented, rows, cols);
    }

    args.modules.optimizer.row_inv_norm(
        args.stream,
        &args.scratch.oriented,
        &mut args.scratch.row_scale,
        rows,
        cols,
        EPS,
    )?;
    for iter in 0..PP_ITERATIONS {
        args.modules.optimizer.row_scale_apply(
            args.stream,
            &args.scratch.oriented,
            &args.scratch.row_scale,
            &mut args.scratch.scaled,
            rows,
            cols,
        )?;
        polar(args, PolarSource::Scaled, rows, cols)?;
        if iter + 1 < PP_ITERATIONS {
            refine_row_scale(args, rows, cols)?;
        }
    }
    Ok(())
}

fn refine_row_scale(
    args: &mut AuroraMatrixArgs<'_, '_>,
    rows: u32,
    cols: u32,
) -> Result<(), DriverError> {
    args.modules.optimizer.row_scale_refine(
        args.stream,
        &args.scratch.u,
        &mut args.scratch.row_scale,
        rows,
        cols,
        cols as f32 / rows as f32,
        EPS,
    )
}
