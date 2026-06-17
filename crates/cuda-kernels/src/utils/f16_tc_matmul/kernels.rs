use cuda_device::{DisjointSlice, cuda_module, kernel};

use super::convert::fp32_to_f16_body;
use super::matmul::matmul_body;
use super::pad::pad_rows_body;

pub const F16_THREADS_PER_BLOCK: u32 = 256;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn f16_fp32_pad_rows_kernel(
        src: &[f32],
        dst: DisjointSlice<f32>,
        rows: u32,
        src_cols: u32,
        dst_cols: u32,
    ) {
        pad_rows_body(src, dst, rows, src_cols, dst_cols);
    }

    #[kernel]
    pub fn fp32_to_f16_kernel(src: &[f32], dst: DisjointSlice<u16>, element_count: u32) {
        fp32_to_f16_body(src, dst, element_count);
    }

    #[kernel]
    pub fn f16_batched_tc_matmul_kernel(
        a: &[u16],
        b_t: &[u16],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        matmul_body(a, b_t, out, batch_count, m, n, k);
    }
}

pub(crate) use module::LoadedModule;
