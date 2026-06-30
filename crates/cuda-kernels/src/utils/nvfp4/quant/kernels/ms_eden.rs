use cuda_device::{DisjointSlice, cuda_module, kernel, warp};

use crate::nvfp4::nvfp4_rowwise_value;

const HADAMARD_DIM: u32 = 32;
const GROUP_SIZE: u32 = 16;
const INV_SQRT_32: f32 = 0.176_776_69;
const FP4_MAX: f32 = 6.0;
const AMAX_WARPS_PER_BLOCK: u32 = crate::nvfp4_quant::config::WARPS_PER_BLOCK;

#[path = "ms_eden/amax.rs"]
pub(crate) mod amax;
#[path = "ms_eden/body.rs"]
mod body;
#[path = "ms_eden/fp32.rs"]
pub(crate) mod fp32;
#[path = "ms_eden/fp32_pair.rs"]
pub(crate) mod fp32_pair;
#[path = "ms_eden/fp32_transpose.rs"]
pub(crate) mod fp32_transpose;
#[path = "ms_eden/input.rs"]
mod input;
#[path = "ms_eden/nvfp4_transpose.rs"]
pub(crate) mod nvfp4_transpose;
#[path = "ms_eden/pack.rs"]
mod pack;
#[path = "ms_eden/random.rs"]
mod random;

#[allow(static_mut_refs)]
#[cuda_module]
pub(crate) mod module {
    use super::input::*;
    use super::pack::*;
    use super::random::random_sign;
    use super::*;

    macro_rules! guarded_pack_chunk {
        ($chunk:ident, $chunk_count:ident) => {
            let $chunk = pack_chunk();
            if $chunk >= $chunk_count {
                return;
            }
        };
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut out_chunk_amax: DisjointSlice<f32>,
        global_scale: &[f32],
        chunk_count: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        guarded_pack_chunk!(chunk, chunk_count);

        let chunk_base = chunk * HADAMARD_DIM;
        let input = rowwise_transposed_hadamard_input(
            bytes,
            scales,
            source_global_scales,
            chunk_base,
            lane,
            source_rows,
            source_cols,
            dst_row_len,
            sign_seed,
        );
        ms_eden_pack_chunk(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            &mut out_chunk_amax,
            chunk,
            dst_row_len,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        chunk_count: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        guarded_pack_chunk!(chunk, chunk_count);

        let chunk_base = chunk * HADAMARD_DIM;
        let input = rowwise_transposed_hadamard_input(
            bytes,
            scales,
            source_global_scales,
            chunk_base,
            lane,
            source_rows,
            source_cols,
            dst_row_len,
            sign_seed,
        );
        ms_eden_pack_chunk_no_chunk_amax(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            dst_row_len,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk = pack_chunk();
        let chunk_base = chunk * HADAMARD_DIM;
        let input = rowwise_transposed_hadamard_input(
            bytes,
            scales,
            source_global_scales,
            chunk_base,
            lane,
            source_rows,
            source_cols,
            dst_row_len,
            sign_seed,
        );
        ms_eden_pack_chunk_no_chunk_amax(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            dst_row_len,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_cols: u32,
        chunks_per_row_shift: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk = pack_chunk();
        let chunk_in_row_mask = (1u32 << chunks_per_row_shift) - 1;
        let row = chunk >> chunks_per_row_shift;
        let chunk_in_row = (chunk & chunk_in_row_mask) * HADAMARD_DIM;
        let input_col = chunk_in_row + lane;
        let input = nvfp4_rowwise_value(
            bytes,
            scales,
            source_global_scales,
            source_cols as usize,
            input_col as usize,
            row as usize,
        ) * random_sign(sign_seed, input_col);
        ms_eden_pack_chunk_no_chunk_amax_row(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            row,
            chunk_in_row == 0,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_source_cols_pow2_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_cols_shift: u32,
        chunks_per_row_shift: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk = pack_chunk();
        let chunk_in_row_mask = (1u32 << chunks_per_row_shift) - 1;
        let row = chunk >> chunks_per_row_shift;
        let chunk_in_row = (chunk & chunk_in_row_mask) * HADAMARD_DIM;
        let input_col = chunk_in_row + lane;
        let input = nvfp4_rowwise_value_at_pow2(
            bytes,
            scales,
            source_global_scales,
            source_cols_shift,
            input_col,
            row,
        ) * random_sign(sign_seed, input_col);
        ms_eden_pack_chunk_no_chunk_amax_row(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            row,
            chunk_in_row == 0,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }
}
