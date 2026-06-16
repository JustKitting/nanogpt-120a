use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, ptx_asm, thread};

const EMBEDDING_THREADS_PER_BLOCK: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmbeddingParams {
    pub hidden_len: u32,
    pub embedding_dim: u32,
    pub token_embedding_global_scale: f32,
    pub position_embedding_global_scale: f32,
}

unsafe impl DeviceCopy for EmbeddingParams {}

pub struct Nvfp4DeviceTensor<'a> {
    pub bytes: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub global_scale: f32,
}

pub struct EmbeddingArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub tokens: &'a DeviceBuffer<u32>,
    pub token_embedding: Nvfp4DeviceTensor<'a>,
    pub position_embedding: Nvfp4DeviceTensor<'a>,
    pub hidden: &'out mut DeviceBuffer<f32>,
    pub hidden_len: u32,
    pub embedding_dim: u32,
}

impl<'a, 'out> EmbeddingArgs<'a, 'out> {
    pub fn new(
        stream: &'a CudaStream,
        tokens: &'a DeviceBuffer<u32>,
        token_embedding: Nvfp4DeviceTensor<'a>,
        position_embedding: Nvfp4DeviceTensor<'a>,
        hidden: &'out mut DeviceBuffer<f32>,
        hidden_len: u32,
        embedding_dim: u32,
    ) -> Self {
        Self {
            stream,
            tokens,
            token_embedding,
            position_embedding,
            hidden,
            hidden_len,
            embedding_dim,
        }
    }
}

pub struct EmbeddingModule {
    module: kernels::LoadedModule,
}

impl EmbeddingModule {
    pub fn from_module(module: CudaModule) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module.into())?,
        })
    }

    pub fn token_position_embedding(&self, args: EmbeddingArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.token_position_embedding_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.hidden_len.div_ceil(EMBEDDING_THREADS_PER_BLOCK), 1, 1),
                block_dim: (EMBEDDING_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.tokens,
            args.token_embedding.bytes,
            args.token_embedding.scales,
            args.position_embedding.bytes,
            args.position_embedding.scales,
            args.hidden,
            EmbeddingParams {
                hidden_len: args.hidden_len,
                embedding_dim: args.embedding_dim,
                token_embedding_global_scale: args.token_embedding.global_scale,
                position_embedding_global_scale: args.position_embedding.global_scale,
            },
        )
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn token_position_embedding_kernel(
        tokens: &[u32],
        token_embedding_bytes: &[u8],
        token_embedding_scales: &[u8],
        position_embedding_bytes: &[u8],
        position_embedding_scales: &[u8],
        mut hidden: DisjointSlice<f32>,
        params: EmbeddingParams,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();

        if index < params.hidden_len {
            let pos = index / params.embedding_dim;
            let dim = index - pos * params.embedding_dim;
            let token = tokens[pos as usize];
            let token_index = token as usize * params.embedding_dim as usize + dim as usize;
            let position_index = pos as usize * params.embedding_dim as usize + dim as usize;

            let token_value = nvfp4_value(
                token_embedding_bytes,
                token_embedding_scales,
                params.token_embedding_global_scale,
                token_index,
            );
            let position_value = nvfp4_value(
                position_embedding_bytes,
                position_embedding_scales,
                params.position_embedding_global_scale,
                position_index,
            );

            unsafe {
                *hidden.get_unchecked_mut(index as usize) = token_value + position_value;
            }
        }
    }

    #[inline(always)]
    fn nvfp4_value(bytes: &[u8], scales: &[u8], global_scale: f32, index: usize) -> f32 {
        let byte = bytes[index / 2];
        let payload = if index & 1 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        };

        e2m1_value(payload) * e4m3_value(scales[index / 16] as u16) * global_scale
    }

    #[inline(always)]
    fn e2m1_value(bits: u8) -> f32 {
        let value: f32;
        let packed = bits as u16;

        unsafe {
            ptx_asm!(
                "{ .reg .b8 e2; .reg .b32 h2; .reg .b16 lo; cvt.u8.u16 e2, %1; cvt.rn.f16x2.e2m1x2 h2, e2; cvt.u16.u32 lo, h2; cvt.f32.f16 %0, lo; }",
                out("=f") value,
                in("h") packed,
                options(register_only),
            );
        }
        value
    }

    #[inline(always)]
    fn e4m3_value(bits: u16) -> f32 {
        let value: f32;

        unsafe {
            ptx_asm!(
                "{ .reg .b32 h2; .reg .b16 lo; cvt.rn.f16x2.e4m3x2 h2, %1; cvt.u16.u32 lo, h2; cvt.f32.f16 %0, lo; }",
                out("=f") value,
                in("h") bits,
                options(register_only),
            );
        }
        value
    }
}
