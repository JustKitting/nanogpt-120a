use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantModule, RowAmaxArgs};

use crate::types::HiddenStateNvfp4;

pub(super) fn requantize_attention(
    quant_module: &Nvfp4QuantModule,
    stream: &CudaStream,
    mut input_nvfp4: HiddenStateNvfp4<'_>,
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

    input_nvfp4.quantize_precomputed_amax(
        quant_module,
        stream,
        normalized,
        normalized_amax,
        row_count,
        crate::GPT2_N_EMBD as u32,
    )
}
