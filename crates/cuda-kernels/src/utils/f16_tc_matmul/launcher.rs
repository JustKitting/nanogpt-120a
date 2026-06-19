use std::sync::Arc;

use cuda_core::{CudaModule, DriverError, LaunchConfig};

use super::args::{F16TcMatmulArgs, F16TcMatmulF32Args};
use super::cta_tile::{CTA_M, CTA_N, CTA_THREADS};
use super::kernels;
use super::prepare::prepare_halves;

pub struct F16TcMatmulModule {
    pub(super) module: kernels::module::LoadedModule,
}

impl F16TcMatmulModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::module::from_module(module)?,
        })
    }

    pub fn batched_matmul(&self, args: F16TcMatmulArgs<'_, '_, '_>) -> Result<(), DriverError> {
        assert!(args.out.len() >= args.batch_count as usize * args.m as usize * args.n as usize);

        let (scratch, k) = prepare_halves(
            &self.module,
            args.stream,
            args.a,
            args.b_t,
            args.scratch,
            args.batch_count,
            args.m,
            args.n,
            args.k,
        )?;

        self.module.f16_cta_tc_matmul_kernel(
            args.stream,
            cta_config(args.m, args.n, args.batch_count),
            scratch.a_halves,
            scratch.b_t_halves,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            k,
        )
    }

    pub fn batched_matmul_f32_input(
        &self,
        args: F16TcMatmulF32Args<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.a.len() >= args.batch_count as usize * args.m as usize * args.k as usize);
        assert!(args.b_t.len() >= args.batch_count as usize * args.n as usize * args.k as usize);
        assert!(args.out.len() >= args.batch_count as usize * args.m as usize * args.n as usize);

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
}

pub(super) fn cta_config(m: u32, n: u32, batch_count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(CTA_N), m.div_ceil(CTA_M), batch_count),
        block_dim: (CTA_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}
