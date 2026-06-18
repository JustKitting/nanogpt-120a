use std::sync::Arc;

use cuda_core::{CudaModule, DriverError, LaunchConfig};

use super::args::{F16TcMatmulAddArgs, F16TcMatmulArgs, F16TcSymmetricMatmulArgs};
use super::kernels;
use super::prepare::{prepare_halves, prepare_self_halves};

pub struct F16TcMatmulModule {
    module: kernels::module::LoadedModule,
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

        self.module.f16_batched_tc_matmul_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.n.div_ceil(8), args.m.div_ceil(16), args.batch_count),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            scratch.a_halves,
            scratch.b_t_halves,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            k,
        )
    }

    pub fn batched_matmul_add(
        &self,
        args: F16TcMatmulAddArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.base.len() >= args.batch_count as usize * args.m as usize * args.n as usize);
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

        self.module.f16_batched_tc_matmul_add_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.n.div_ceil(8), args.m.div_ceil(16), args.batch_count),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            scratch.a_halves,
            scratch.b_t_halves,
            args.base,
            args.out,
            args.batch_count,
            args.m,
            args.n,
            k,
            args.base_scale,
            args.matmul_scale,
        )
    }

    pub fn symmetric_matmul(
        &self,
        args: F16TcSymmetricMatmulArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.out.len() >= args.rows as usize * args.rows as usize);
        let (scratch, cols) = prepare_self_halves(
            &self.module,
            args.stream,
            args.x,
            args.scratch,
            args.rows,
            args.cols,
        )?;
        self.module.f16_symmetric_tc_matmul_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.rows.div_ceil(8), args.rows.div_ceil(16), 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            scratch.a_halves,
            args.out,
            args.rows,
            cols,
        )
    }
}
