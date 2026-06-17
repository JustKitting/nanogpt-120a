use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantModule, Nvfp4QuantRowwiseArgs, RowAmaxArgs};

use crate::types::HiddenStateNvfp4;

pub(super) fn requantize_attention(
    quant_module: &Nvfp4QuantModule,
    stream: &CudaStream,
    input_nvfp4: HiddenStateNvfp4<'_>,
    normalized: &DeviceBuffer<f32>,
    normalized_amax: &mut DeviceBuffer<f32>,
    row_count: u32,
) -> Result<(), DriverError> {
    quant_module.row_amax_f32(RowAmaxArgs {
        stream,
        x: normalized,
        out: normalized_amax,
        row_count,
        row_len: crate::GPT2_N_EMBD as u32,
    })?;

    quant_module.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
        stream,
        x: normalized,
        amax: normalized_amax,
        out_fp4: &mut *input_nvfp4.bytes,
        out_scales: &mut *input_nvfp4.scales,
        out_global_scale: &mut *input_nvfp4.global_scales,
        group_count: row_count * crate::GPT2_N_EMBD as u32 / 16,
        row_len: crate::GPT2_N_EMBD as u32,
    })
}
