use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantModule, RowAmaxArgs};

use crate::types::MlpActivationNvfp4;

pub(super) fn quantize_activation(
    quant_module: &Nvfp4QuantModule,
    stream: &CudaStream,
    activation: &DeviceBuffer<f32>,
    mut activation_nvfp4: MlpActivationNvfp4<'_>,
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

    activation_nvfp4.quantize_precomputed_amax(
        quant_module,
        stream,
        activation,
        normalized_amax,
        row_count,
        crate::GPT2_MLP as u32,
    )
}
