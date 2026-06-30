use cuda_core::DriverError;

use super::args::{
    F16TcMatmulF32ATransposedHalfRhsArgs, F16TcMatmulF32ATransposedRhsArgs, F16TcMatmulF32Args,
    F16TcMatmulF32HalfRhsArgs, F16TcMatmulF32RhsArgs,
};
use super::launcher::{F16TcMatmulModule, cta_config, elements};

impl F16TcMatmulModule {
    pub fn batched_matmul_f32_input(
        &self,
        args: F16TcMatmulF32Args<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.b_t.len() >= elements(args.batch_count, args.n, args.k));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        self.module.f16_cta_tc_matmul_f32_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            args.a,
            args.b_t,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
        )
    }

    pub fn batched_matmul_f32_rhs(
        &self,
        args: F16TcMatmulF32RhsArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.rhs.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        self.module.f16_cta_tc_matmul_f32_rhs_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            args.a,
            args.rhs,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
        )
    }

    pub fn batched_matmul_f32_half_rhs(
        &self,
        args: F16TcMatmulF32HalfRhsArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.rhs.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        self.module.f16_cta_tc_matmul_f32_half_rhs_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            args.a,
            args.rhs,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
        )
    }

    pub fn batched_matmul_f32_a_transposed_rhs(
        &self,
        args: F16TcMatmulF32ATransposedRhsArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.a.len() >= elements(args.batch_count, args.k, args.m));
        assert!(args.rhs.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        self.module.f16_cta_tc_matmul_f32_a_transposed_rhs_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            args.a,
            args.rhs,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
        )
    }

    pub fn batched_matmul_f32_a_transposed_half_rhs(
        &self,
        args: F16TcMatmulF32ATransposedHalfRhsArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.a.len() >= elements(args.batch_count, args.k, args.m));
        assert!(args.rhs.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        self.module
            .f16_cta_tc_matmul_f32_a_transposed_half_rhs_kernel(
                args.stream,
                cta_config(args.m, args.n, args.batch_count),
                args.a,
                args.rhs,
                args.out,
                args.batch_count,
                args.m,
                args.n,
                args.k,
            )
    }
}
