use cuda_core::DriverError;
use rust_kernels_cuda::f16_tc_matmul::{
    F16TcMatmulAddRhsTransposeBaseArgs, F16TcMatmulArgs, F16TcMatmulScratch,
};

use super::AuroraMatrixArgs;
use super::polar_express_coefficients::polar_express_coefficients;

pub(super) fn polar_express_iteration(
    args: &mut AuroraMatrixArgs<'_, '_>,
    rows: u32,
    cols: u32,
    iter: usize,
) -> Result<(), DriverError> {
    let coefficients = polar_express_coefficients(iter);

    args.modules.f16_tc.batched_matmul(F16TcMatmulArgs {
        stream: args.stream,
        a: &args.scratch.polar_x,
        b_t: &args.scratch.polar_x,
        out: &mut args.scratch.polar_gram,
        scratch: F16TcMatmulScratch {
            a_padded: &mut args.scratch.a_padded,
            b_t_padded: &mut args.scratch.b_t_padded,
            a_halves: &mut args.scratch.a_halves,
            b_t_halves: &mut args.scratch.b_t_halves,
        },
        batch_count: 1,
        m: rows,
        n: rows,
        k: cols,
    })?;

    args.modules.f16_tc.batched_matmul_add_rhs_transposed_base(
        F16TcMatmulAddRhsTransposeBaseArgs {
            stream: args.stream,
            a: &args.scratch.polar_gram,
            rhs: &args.scratch.polar_x,
            base: &args.scratch.polar_x,
            out: &mut args.scratch.polar_ax,
            batch_count: 1,
            m: rows,
            n: cols,
            k: rows,
            base_scale: 0.0,
            matmul_scale: 1.0,
        },
    )?;

    args.modules.optimizer.elementwise_linear_combination(
        args.stream,
        &args.scratch.polar_x,
        &args.scratch.polar_ax,
        &mut args.scratch.u,
        coefficients.a,
        coefficients.b,
        rows * cols,
    )?;

    args.modules.f16_tc.batched_matmul_add_rhs_transposed_base(
        F16TcMatmulAddRhsTransposeBaseArgs {
            stream: args.stream,
            a: &args.scratch.polar_gram,
            rhs: &args.scratch.polar_ax,
            base: &args.scratch.u,
            out: &mut args.scratch.oriented,
            batch_count: 1,
            m: rows,
            n: cols,
            k: rows,
            base_scale: 1.0,
            matmul_scale: coefficients.c,
        },
    )?;
    std::mem::swap(&mut args.scratch.polar_x, &mut args.scratch.oriented);
    Ok(())
}
