use cuda_core::DriverError;

use super::args::{F16TcMatmulAddArgs, F16TcMatmulAddRhsTransposeArgs};
use super::launcher::{F16TcMatmulModule, cta_config};

impl F16TcMatmulModule {
    pub fn batched_matmul_add(
        &self,
        args: F16TcMatmulAddArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.a.len() >= elements(args.batch_count, args.m, args.k));
        assert!(args.b_t.len() >= elements(args.batch_count, args.n, args.k));
        assert!(args.base.len() >= elements(args.batch_count, args.m, args.n));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        self.module.f16_cta_tc_matmul_add_f32_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            args.a,
            args.b_t,
            args.base,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
            args.base_scale,
            args.matmul_scale,
        )
    }

    pub fn batched_matmul_add_rhs_transposed(
        &self,
        args: F16TcMatmulAddRhsTransposeArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.rhs_base.len() >= elements(args.batch_count, args.k, args.n));
        assert!(args.out.len() >= elements(args.batch_count, args.m, args.n));

        self.module.f16_cta_tc_matmul_add_f32_rhs_transposed_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            args.a,
            args.rhs_base,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            args.k,
            args.base_scale,
            args.matmul_scale,
        )
    }
}

fn elements(batch_count: u32, rows: u32, cols: u32) -> usize {
    batch_count as usize * rows as usize * cols as usize
}
