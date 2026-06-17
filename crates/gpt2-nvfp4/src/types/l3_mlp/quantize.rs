use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantModule, Nvfp4QuantRowwiseArgs, RowAmaxArgs};

use crate::types::MlpActivationNvfp4;

pub(super) fn quantize_activation(
    quant_module: &Nvfp4QuantModule,
    stream: &CudaStream,
    activation: &DeviceBuffer<f32>,
    activation_nvfp4: MlpActivationNvfp4<'_>,
    normalized_amax: &mut DeviceBuffer<f32>,
    row_count: u32,
) -> Result<(), DriverError> {
    quant_module.row_amax_f32(RowAmaxArgs {
        stream,
        x: activation,
        out: normalized_amax,
        row_count,
        row_len: crate::GPT2_MLP as u32,
    })?;

    quant_module.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
        stream,
        x: activation,
        amax: normalized_amax,
        out_fp4: &mut *activation_nvfp4.bytes,
        out_scales: &mut *activation_nvfp4.scales,
        out_global_scale: &mut *activation_nvfp4.global_scales,
        group_count: row_count * crate::GPT2_MLP as u32 / 16,
        row_len: crate::GPT2_MLP as u32,
    })
}
