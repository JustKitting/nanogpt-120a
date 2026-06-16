use std::marker::PhantomData;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::kernel_config::TransformerKernelConfig;

const ATTENTION_THREADS_PER_BLOCK: u32 = 256;

pub struct FakeAttentionArgs<'a, 'out, C: TransformerKernelConfig> {
    pub stream: &'a CudaStream,
    pub hidden: &'out mut DeviceBuffer<f32>,
    config: PhantomData<C>,
}

impl<'a, 'out, C: TransformerKernelConfig> FakeAttentionArgs<'a, 'out, C> {
    pub fn new(stream: &'a CudaStream, hidden: &'out mut DeviceBuffer<f32>) -> Self {
        Self {
            stream,
            hidden,
            config: PhantomData,
        }
    }
}

pub struct AttentionModule {
    module: kernels::LoadedModule,
}

impl AttentionModule {
    pub fn from_module(module: CudaModule) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module.into())?,
        })
    }

    pub fn fake_attention<C: TransformerKernelConfig>(
        &self,
        args: FakeAttentionArgs<'_, '_, C>,
    ) -> Result<(), DriverError> {
        self.module.fake_attention_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (C::HIDDEN_LEN.div_ceil(ATTENTION_THREADS_PER_BLOCK), 1, 1),
                block_dim: (ATTENTION_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.hidden,
            C::HIDDEN_LEN,
        )
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn fake_attention_kernel(hidden: DisjointSlice<f32>, hidden_len: u32) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();

        if index < hidden_len {
            let _ = hidden.len();
        }
    }
}
