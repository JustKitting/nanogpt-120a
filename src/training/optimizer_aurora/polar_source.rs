use cuda_core::DriverError;
use rust_kernels_cuda::transpose::TransposeF32Args;

use super::AuroraMatrixArgs;
use super::polar::PolarSource;

pub(super) fn transpose_source_to_polar_x(
    args: &mut AuroraMatrixArgs<'_, '_>,
    source: PolarSource,
    rows: u32,
    cols: u32,
) -> Result<(), DriverError> {
    match source {
        PolarSource::Oriented => args.modules.transpose.transpose_f32(TransposeF32Args {
            stream: args.stream,
            input: &args.scratch.oriented,
            output: &mut args.scratch.polar_x,
            rows,
            cols,
        }),
        PolarSource::Scaled => args.modules.transpose.transpose_f32(TransposeF32Args {
            stream: args.stream,
            input: &args.scratch.scaled,
            output: &mut args.scratch.polar_x,
            rows,
            cols,
        }),
    }
}

pub(super) fn normalize_source_to_polar_x(
    args: &mut AuroraMatrixArgs<'_, '_>,
    source: PolarSource,
    len: u32,
) -> Result<(), DriverError> {
    match source {
        PolarSource::Oriented => args.modules.optimizer.polar_normalize(
            args.stream,
            &args.scratch.oriented,
            &mut args.scratch.polar_x,
            &mut args.scratch.polar_chunks,
            &mut args.scratch.polar_inv_norm,
            len,
        ),
        PolarSource::Scaled => args.modules.optimizer.polar_normalize(
            args.stream,
            &args.scratch.scaled,
            &mut args.scratch.polar_x,
            &mut args.scratch.polar_chunks,
            &mut args.scratch.polar_inv_norm,
            len,
        ),
    }
}
