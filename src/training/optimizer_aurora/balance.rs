use cuda_core::DriverError;

use super::polar::{PolarSource, polar};
use super::{AuroraMatrixArgs, EPS};

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
    args.modules.optimizer.row_scale_apply(
        args.stream,
        &args.scratch.oriented,
        &args.scratch.row_scale,
        &mut args.scratch.scaled,
        rows,
        cols,
    )?;
    polar(args, PolarSource::Scaled, rows, cols)
}
