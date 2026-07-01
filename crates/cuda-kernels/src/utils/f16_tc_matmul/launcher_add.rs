use cuda_core::DriverError;

use super::args::{F16TcMatmulAddArgs, F16TcMatmulAddRhsTransposeBaseArgs};
use super::launcher::{F16TcMatmulModule, cta_config, elements};

macro_rules! add_launcher {
    ($method:ident, $args:ty, $rhs:ident, $kernel:ident, rhs($rhs_rows:ident, $rhs_cols:ident)) => {
        pub fn $method(&self, args: $args) -> Result<(), DriverError> {
            assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
            assert!(args.$rhs.len() >= elements(args.batch_count, args.$rhs_rows, args.$rhs_cols));
            assert!(args.base.len() >= elements(args.batch_count, args.m, args.n));
            assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));
            self.module.$kernel(
                args.stream, cta_config(args.m, args.n, args.batch_count), args.a, args.$rhs,
                args.base, args.out, args.batch_count, args.m, args.n, args.k,
                args.base_scale, args.matmul_scale,
            )
        }
    };
}

impl F16TcMatmulModule {
    add_launcher!(batched_matmul_add, F16TcMatmulAddArgs<'_, '_, '_>, b_t, f16_cta_tc_matmul_add_f32_kernel, rhs(n, k));
    add_launcher!(batched_matmul_add_rhs_transposed_base, F16TcMatmulAddRhsTransposeBaseArgs<'_, '_>, rhs, f16_cta_tc_matmul_add_f32_rhs_transposed_base_kernel, rhs(k, n));
}
