use cuda_core::DriverError;
use rust_kernels_cuda::transpose::TransposeF32Args;

use super::AuroraMatrixArgs;

pub(super) fn orient_update(
    args: &mut AuroraMatrixArgs<'_, '_>,
) -> Result<(u32, u32, bool), DriverError> {
    if args.rows < args.cols {
        args.modules.transpose.transpose_f32(TransposeF32Args {
            stream: args.stream,
            input: &args.scratch.update,
            output: &mut args.scratch.oriented,
            rows: args.rows,
            cols: args.cols,
        })?;
        return Ok((args.cols, args.rows, true));
    }

    args.modules.optimizer.matrix_combine(
        args.stream,
        &args.scratch.update,
        &args.scratch.update,
        &mut args.scratch.oriented,
        1.0,
        0.0,
        args.rows * args.cols,
    )?;
    Ok((args.rows, args.cols, false))
}
