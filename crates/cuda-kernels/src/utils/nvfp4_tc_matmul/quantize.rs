use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::args::{Nvfp4TcMatmulOperand, QUARTET_MS_EDEN_SCALE_OVERRIDE};
use crate::nvfp4_quant::{MsEdenQuantArgs, Nvfp4QuantModule};

pub(super) fn quantize_operand(
    module: &Nvfp4QuantModule,
    stream: &CudaStream,
    x: &DeviceBuffer<f32>,
    scratch: &mut Nvfp4TcMatmulOperand<'_>,
    row_count: u32,
    row_len: u32,
    sign_seed: u32,
    scale_seed: u32,
) -> Result<(), DriverError> {
    module.fp32_to_nvfp4_ms_eden(MsEdenQuantArgs {
        stream,
        x,
        out_fp4: scratch.bytes,
        out_scales: scratch.scales,
        out_global_scales: scratch.global_scales,
        out_chunk_amax: scratch.chunk_amax,
        row_count,
        row_len,
        global_scale: scratch.global_scale,
        scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
        sign_seed,
        scale_seed,
    })
}
