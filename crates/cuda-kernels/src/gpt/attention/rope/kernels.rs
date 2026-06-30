use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use super::{ApplyRopeParams, THREADS_PER_BLOCK};
use crate::attention::layout::rope_qkv_index;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::float_ptx::{exp_f32, fma_f32, sincos_f32};

pub use module::{LoadedModule, from_module};

#[cuda_module]
pub mod module {
    use super::*;

    #[kernel]
    pub fn apply_rope_kernel(mut qkv: DisjointSlice<f32>, params: ApplyRopeParams) {
        let Some((batch, token, head, dim)) = rope_position(&params) else {
            return;
        };
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

    #[kernel]
    pub fn apply_rope_save_f16_kernel(
        mut qkv: DisjointSlice<f32>,
        mut qkv_f16: DisjointSlice<u16>,
        params: ApplyRopeParams,
    ) {
        let Some((batch, token, head, dim)) = rope_position(&params) else {
            return;
        };
        let q = rotate_section(&mut qkv, batch, token, head, dim, 0, &params);
        let k = rotate_section(
            &mut qkv,
            batch,
            token,
            head,
            dim,
            params.embedding_dim,
            &params,
        );
        let v = read_pair(
            &mut qkv,
            batch,
            token,
            head,
            dim,
            params.embedding_dim * 2,
            &params,
        );

        store_pair_f16(&mut qkv_f16, q);
        store_pair_f16(&mut qkv_f16, k);
        store_pair_f16(&mut qkv_f16, v);
    }

    #[inline(always)]
    fn rope_position(params: &ApplyRopeParams) -> Option<(u32, u32, u32, u32)> {
        let half_head_dim = params.head_dim / 2;
        let index = thread::blockIdx_x() * THREADS_PER_BLOCK + thread::threadIdx_x();
        let total = params.batch_size * params.seq_len * params.head_count * half_head_dim;
        if index >= total {
            return None;
        }

        let pair = index % half_head_dim;
        let head = (index / half_head_dim) % params.head_count;
        let token = (index / (half_head_dim * params.head_count)) % params.seq_len;
        let batch = index / (half_head_dim * params.head_count * params.seq_len);
        if batch * params.seq_len + token >= params.row_count {
            return None;
        }

        Some((batch, token, head, pair * 2))
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
    ) -> QkvPair {
        let even_index = rope_qkv_index(batch, token, head, dim, section_offset, params);
        let odd_index = rope_qkv_index(batch, token, head, dim + 1, section_offset, params);
        let ptr = qkv.as_mut_ptr();
        let even = unsafe { *ptr.add(even_index) };
        let odd = unsafe { *ptr.add(odd_index) };
        let (sin, cos) = sincos_f32(token as f32 * rope_inv_freq(dim, params.head_dim));
        let rotated_even = fma_f32(-odd, sin, even * cos);
        let rotated_odd = fma_f32(odd, cos, even * sin);

        unsafe {
            *ptr.add(even_index) = rotated_even;
            *ptr.add(odd_index) = rotated_odd;
        }

        QkvPair {
            even_index,
            even: rotated_even,
            odd_index,
            odd: rotated_odd,
        }
    }

    #[inline(always)]
    fn read_pair(
        qkv: &mut DisjointSlice<f32>,
        batch: u32,
        token: u32,
        head: u32,
        dim: u32,
        section_offset: u32,
        params: &ApplyRopeParams,
    ) -> QkvPair {
        let even_index = rope_qkv_index(batch, token, head, dim, section_offset, params);
        let odd_index = rope_qkv_index(batch, token, head, dim + 1, section_offset, params);
        let ptr = qkv.as_mut_ptr();

        QkvPair {
            even_index,
            even: unsafe { *ptr.add(even_index) },
            odd_index,
            odd: unsafe { *ptr.add(odd_index) },
        }
    }

    #[inline(always)]
    fn store_pair_f16(qkv_f16: &mut DisjointSlice<u16>, pair: QkvPair) {
        unsafe {
            *qkv_f16.get_unchecked_mut(pair.even_index) = cvt_rn_f16_f32(pair.even);
            *qkv_f16.get_unchecked_mut(pair.odd_index) = cvt_rn_f16_f32(pair.odd);
        }
    }

    #[derive(Clone, Copy)]
    struct QkvPair {
        even_index: usize,
        even: f32,
        odd_index: usize,
        odd: f32,
    }

    #[inline(always)]
    fn rope_inv_freq(dim: u32, head_dim: u32) -> f32 {
        exp_f32(-9.210_340_5 * dim as f32 / head_dim as f32)
    }
}
