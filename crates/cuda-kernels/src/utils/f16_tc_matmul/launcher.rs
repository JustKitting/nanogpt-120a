use std::sync::Arc;

use cuda_core::{CudaModule, DriverError, LaunchConfig};

use super::args::{F16TcMatmulArgs, f16_tc_matmul_padded_k};
use super::kernels;
use super::launch_ops::{convert, pad_rows};

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
        let padded_k = f16_tc_matmul_padded_k(args.k);
        let a_rows = args.batch_count * args.m;
        let b_rows = args.batch_count * args.n;
        assert!(args.a.len() >= a_rows as usize * args.k as usize);
        assert!(args.b_t.len() >= b_rows as usize * args.k as usize);
        assert!(args.out.len() >= args.batch_count as usize * args.m as usize * args.n as usize);

        let scratch = args.scratch;
        assert!(scratch.a_padded.len() >= a_rows as usize * padded_k as usize);
        assert!(scratch.b_t_padded.len() >= b_rows as usize * padded_k as usize);
        assert!(scratch.a_halves.len() >= a_rows as usize * padded_k as usize);
        assert!(scratch.b_t_halves.len() >= b_rows as usize * padded_k as usize);

        pad_rows(
            &self.module,
            args.stream,
            args.a,
            scratch.a_padded,
            a_rows,
            args.k,
            padded_k,
        )?;
        pad_rows(
            &self.module,
            args.stream,
            args.b_t,
            scratch.b_t_padded,
            b_rows,
            args.k,
            padded_k,
        )?;
        convert(
            &self.module,
            args.stream,
            scratch.a_padded,
            scratch.a_halves,
            a_rows * padded_k,
        )?;
        convert(
            &self.module,
            args.stream,
            scratch.b_t_padded,
            scratch.b_t_halves,
            b_rows * padded_k,
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
            padded_k,
        )
    }
}
