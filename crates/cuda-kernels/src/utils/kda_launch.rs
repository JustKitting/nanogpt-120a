use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use crate::f16_tc_matmul::{
    F16TcMatmulF32ATransposedRhsArgs, F16TcMatmulF32Args, F16TcMatmulF32RhsArgs, F16TcMatmulModule,
};

pub(crate) const KDA_HEAD_DIM: u32 = 64;
pub(crate) const KDA_CHUNK_SIZE: u32 = 64;
pub(crate) const KDA_DECAY_SCALE: f32 = 0.01;

#[derive(Clone, Copy)]
pub(crate) struct LaunchDims {
    pub(crate) batch_head: u32,
    pub(crate) chunks: u32,
    pub(crate) chunk_batch: u32,
    pub(crate) compact_elems: u32,
    pub(crate) chunk_matrix_elems: u32,
    chunk_size: u32,
    head_dim: u32,
}

impl LaunchDims {
    pub(crate) fn new(
        batch_size: u32,
        head_count: u32,
        seq_len: u32,
        head_dim: u32,
        chunk_size: u32,
    ) -> Self {
        let batch_head = batch_size * head_count;
        let chunks = seq_len.div_ceil(chunk_size);
        let chunk_batch = batch_head * chunks;
        Self {
            batch_head,
            chunks,
            chunk_batch,
            compact_elems: batch_head * seq_len * head_dim,
            chunk_matrix_elems: chunk_batch * chunk_size * chunk_size,
            chunk_size,
            head_dim,
        }
    }

    pub(crate) fn cch(self) -> MatmulShape {
        shape(self.chunk_size, self.chunk_size, self.head_dim)
    }

    pub(crate) fn chc(self) -> MatmulShape {
        shape(self.chunk_size, self.head_dim, self.chunk_size)
    }

    pub(crate) fn ccc(self) -> MatmulShape {
        shape(self.chunk_size, self.chunk_size, self.chunk_size)
    }
}

pub(crate) fn linear_config(element_count: u32, threads_per_block: u32) -> LaunchConfig {
    let blocks = element_count.div_ceil(threads_per_block);
    launch_config((blocks, 1, 1), threads_per_block)
}

pub(crate) fn chunk_dim_config(
    batch_head: u32,
    chunks: u32,
    threads_per_block: u32,
) -> LaunchConfig {
    launch_config((batch_head, chunks, 1), threads_per_block)
}

pub(crate) fn matrix_config(batch_count: u32, threads_per_block: u32) -> LaunchConfig {
    launch_config((batch_count, 1, 1), threads_per_block)
}

pub(crate) fn batch_head_config(batch_head: u32, threads_per_block: u32) -> LaunchConfig {
    launch_config((batch_head, 1, 1), threads_per_block)
}

fn launch_config(grid_dim: (u32, u32, u32), threads_per_block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim,
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MatmulShape(u32, u32, u32);

pub(crate) const fn shape(m: u32, n: u32, k: u32) -> MatmulShape {
    MatmulShape(m, n, k)
}

pub(crate) struct MatmulRunner<'a> {
    stream: &'a CudaStream,
    module: &'a F16TcMatmulModule,
    batch_count: u32,
}

macro_rules! matmul_method {
    ($name:ident, $call:ident, $args:ident, $rhs:ident) => {
        pub(crate) fn $name(
            &self,
            a: &DeviceBuffer<f32>,
            $rhs: &DeviceBuffer<f32>,
            out: &mut DeviceBuffer<f32>,
            shape: MatmulShape,
        ) -> Result<(), DriverError> {
            let MatmulShape(m, n, k) = shape;
            self.module.$call($args {
                stream: self.stream,
                a,
                $rhs,
                out,
                batch_count: self.batch_count,
                m,
                n,
                k,
            })
        }
    };
}

impl<'a> MatmulRunner<'a> {
    pub(crate) fn new(
        stream: &'a CudaStream,
        module: &'a F16TcMatmulModule,
        batch_count: u32,
    ) -> Self {
        Self {
            stream,
            module,
            batch_count,
        }
    }

    matmul_method!(f32_input, batched_matmul_f32_input, F16TcMatmulF32Args, b_t);
    matmul_method!(f32_rhs, batched_matmul_f32_rhs, F16TcMatmulF32RhsArgs, rhs);
    matmul_method!(
        f32_a_transposed_rhs,
        batched_matmul_f32_a_transposed_rhs,
        F16TcMatmulF32ATransposedRhsArgs,
        rhs
    );
}
