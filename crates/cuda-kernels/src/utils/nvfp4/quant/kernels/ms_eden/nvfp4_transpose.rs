use cuda_device::{DisjointSlice, cuda_module, kernel, warp};

use super::HADAMARD_DIM;
use super::input::nvfp4_transposed_hadamard_input;
use super::pack::{
    guarded_pack_chunk, ms_eden_pack_chunk, ms_eden_pack_chunk_no_chunk_amax, pack_chunk,
};

#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scale: &[f32],
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
        let input = nvfp4_transposed_hadamard_input(
            bytes,
            scales,
            source_global_scale,
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
    pub fn nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scale: &[f32],
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
        let input = nvfp4_transposed_hadamard_input(
            bytes,
            scales,
            source_global_scale,
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
    pub fn nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scale: &[f32],
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
        let input = nvfp4_transposed_hadamard_input(
            bytes,
            scales,
            source_global_scale,
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
}
