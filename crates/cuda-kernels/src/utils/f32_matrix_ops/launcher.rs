use std::sync::Arc;

use cuda_core::{CudaModule, DriverError, LaunchConfig};

use super::args::{F32AddScaledIdentityArgs, F32Linear2Args};
use super::kernels;

const F32_OPS_THREADS_PER_BLOCK: u32 = 256;

pub struct F32MatrixOpsModule {
    module: kernels::module::LoadedModule,
}

impl F32MatrixOpsModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::module::from_module(module)?,
        })
    }

    pub fn linear2(&self, args: F32Linear2Args<'_, '_>) -> Result<(), DriverError> {
        assert!(args.a.len() >= args.len as usize);
        assert!(args.b.len() >= args.len as usize);
        assert!(args.out.len() >= args.len as usize);

        self.module.f32_linear2_kernel(
            args.stream,
            linear_config(args.len),
            args.a,
            args.b,
            args.out,
            args.len,
            args.a_scale,
            args.b_scale,
        )
    }

    pub fn add_scaled_identity(
        &self,
        args: F32AddScaledIdentityArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let len = args.dim * args.dim;
        assert!(args.src.len() >= len as usize);
        assert!(args.out.len() >= len as usize);

        self.module.f32_add_scaled_identity_kernel(
            args.stream,
            linear_config(len),
            args.src,
            args.out,
            args.dim,
            args.scale,
        )
    }
}

fn linear_config(element_count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (element_count.div_ceil(F32_OPS_THREADS_PER_BLOCK), 1, 1),
        block_dim: (F32_OPS_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
