use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

use super::args::{Nvfp4TcMatmulArgs, nvfp4_tc_matmul_padded_k};
use super::kernels;
use super::pad::pad_rows;
use super::quantize::quantize_operand;
use crate::launch::launch_config;
use crate::mma::{NVFP4_PROJECTION_THREADS_PER_BLOCK, Nvfp4ProjectionParams, projection_grid_dim};

pub struct Nvfp4TcMatmulModule {
    module: kernels::module::LoadedModule,
}

impl Nvfp4TcMatmulModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::module::from_module(module)?,
        })
    }

    pub fn matmul_ms_eden(&self, args: Nvfp4TcMatmulArgs<'_, '_, '_>) -> Result<(), DriverError> {
        let padded_k = nvfp4_tc_matmul_padded_k(args.k);
        assert!(args.a.len() >= args.m as usize * args.k as usize);
        assert!(args.b_t.len() >= args.n as usize * args.k as usize);
        assert!(args.out.len() >= args.m as usize * args.n as usize);

        let mut scratch = args.scratch;
        assert!(scratch.a_padded.len() >= args.m as usize * padded_k as usize);
        assert!(scratch.b_t_padded.len() >= args.n as usize * padded_k as usize);

        pad_rows(
            &self.module,
            args.stream,
            args.a,
            scratch.a_padded,
            args.m,
            args.k,
            padded_k,
        )?;
        pad_rows(
            &self.module,
            args.stream,
            args.b_t,
            scratch.b_t_padded,
            args.n,
            args.k,
            padded_k,
        )?;

        quantize_operand(
            args.quant_module,
            args.stream,
            &*scratch.a_padded,
            &mut scratch.a,
            args.m,
            padded_k,
            (args.sign_seed, args.scale_seed),
        )?;
        quantize_operand(
            args.quant_module,
            args.stream,
            &*scratch.b_t_padded,
            &mut scratch.b_t,
            args.n,
            padded_k,
            (args.sign_seed, args.scale_seed ^ 0x9e37_79b9),
        )?;

        let a = scratch.a.rowwise();
        self.module.nvfp4_tc_matmul_kernel(
            args.stream,
            launch_config(
                projection_grid_dim(args.m, args.n),
                NVFP4_PROJECTION_THREADS_PER_BLOCK,
            ),
            a.bytes,
            a.scales,
            a.global_scales,
            &*scratch.b_t.bytes,
            &*scratch.b_t.scales,
            args.out,
            Nvfp4ProjectionParams::new(args.m, padded_k, args.n)
                .with_global_scales(scratch.b_t.global_scale, 0.0),
        )
    }
}
