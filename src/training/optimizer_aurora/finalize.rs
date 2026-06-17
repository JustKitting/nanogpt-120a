use cuda_core::DriverError;
use rust_kernels_cuda::transpose::TransposeF32Args;

use super::AuroraMatrixArgs;

pub(super) fn finalize_update(
    args: &mut AuroraMatrixArgs<'_, '_>,
    len: u32,
    transposed: bool,
) -> Result<(), DriverError> {
    let source = if transposed {
        args.modules.transpose.transpose_f32(TransposeF32Args {
            stream: args.stream,
            input: &args.scratch.u,
            output: &mut args.scratch.update,
            rows: args.cols,
            cols: args.rows,
        })?;
        &args.scratch.update
    } else {
        &args.scratch.u
    };
    let ratio = args.rows as f32 / args.cols as f32;
    let scale = if ratio > 1.0 { ratio.sqrt() } else { 1.0 };
    args.modules.optimizer.matrix_combine(
        args.stream,
        source,
        source,
        &mut args.scratch.oriented,
        scale,
        0.0,
        len,
    )
}
