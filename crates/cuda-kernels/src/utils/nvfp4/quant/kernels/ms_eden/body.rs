#![expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]

#[path = "body/padded.rs"]
mod padded;
#[path = "body/row.rs"]
mod row;

pub(super) use padded::{
    fp32_to_nvfp4_ms_eden_body, fp32_to_nvfp4_ms_eden_body_no_chunk_amax,
    fp32_transpose_to_nvfp4_ms_eden_body, fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax,
};
pub(super) use row::{
    fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad,
    fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2,
    fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad,
    fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2,
};
