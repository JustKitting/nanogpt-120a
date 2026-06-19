use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use super::AttentionModule;
use crate::float_ptx::{exp_f32, fma_f32, sincos_f32};

const THREADS_PER_BLOCK: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ApplyRopeParams {
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
}

unsafe impl DeviceCopy for ApplyRopeParams {}

pub struct ApplyRopeArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub qkv: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
}

impl AttentionModule {
    pub fn apply_rope(&self, args: ApplyRopeArgs<'_, '_>) -> Result<(), DriverError> {
        let pair_count = args.batch_size * args.seq_len * args.head_count * (args.head_dim / 2);
        self.rope.apply_rope_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (pair_count.div_ceil(THREADS_PER_BLOCK), 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.qkv,
            ApplyRopeParams {
                row_count: args.row_count,
                seq_len: args.seq_len,
                batch_size: args.batch_size,
                embedding_dim: args.embedding_dim,
                qkv_dim: args.qkv_dim,
                head_count: args.head_count,
                head_dim: args.head_dim,
            },
        )
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn apply_rope_kernel(mut qkv: DisjointSlice<f32>, params: ApplyRopeParams) {
        let half_head_dim = params.head_dim / 2;
        let index = thread::blockIdx_x() * THREADS_PER_BLOCK + thread::threadIdx_x();
        let total = params.batch_size * params.seq_len * params.head_count * half_head_dim;
        if index >= total {
            return;
        }

        let pair = index % half_head_dim;
        let head = (index / half_head_dim) % params.head_count;
        let token = (index / (half_head_dim * params.head_count)) % params.seq_len;
        let batch = index / (half_head_dim * params.head_count * params.seq_len);
        let row = batch * params.seq_len + token;
        if row >= params.row_count {
            return;
        }

        let dim = pair * 2;
        rotate_section(&mut qkv, batch, token, head, dim, 0, &params);
        rotate_section(
            &mut qkv,
            batch,
            token,
            head,
            dim,
            params.embedding_dim,
            &params,
        );
    }

    #[inline(always)]
    fn rotate_section(
        qkv: &mut DisjointSlice<f32>,
        batch: u32,
        token: u32,
        head: u32,
        dim: u32,
        section_offset: u32,
        params: &ApplyRopeParams,
    ) {
        let even_index = qkv_index(batch, token, head, dim, section_offset, params);
        let odd_index = qkv_index(batch, token, head, dim + 1, section_offset, params);
        let ptr = qkv.as_mut_ptr();
        let even = unsafe { *ptr.add(even_index) };
        let odd = unsafe { *ptr.add(odd_index) };
        let (sin, cos) = sincos_f32(token as f32 * rope_inv_freq(dim, params.head_dim));

        unsafe {
            *ptr.add(even_index) = fma_f32(-odd, sin, even * cos);
            *ptr.add(odd_index) = fma_f32(odd, cos, even * sin);
        }
    }

    #[inline(always)]
    fn qkv_index(
        batch: u32,
        token: u32,
        head: u32,
        dim: u32,
        section_offset: u32,
        params: &ApplyRopeParams,
    ) -> usize {
        (batch as usize * params.seq_len as usize + token as usize) * params.qkv_dim as usize
            + section_offset as usize
            + head as usize * params.head_dim as usize
            + dim as usize
    }

    #[inline(always)]
    fn rope_inv_freq(dim: u32, head_dim: u32) -> f32 {
        exp_f32(-9.210_340_5 * dim as f32 / head_dim as f32)
    }
}
