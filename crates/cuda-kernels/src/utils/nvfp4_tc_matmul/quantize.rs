use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::args::Nvfp4TcMatmulOperand;
use crate::nvfp4_quant::{Nvfp4QuantModule, QuartetBackwardMsEdenQuantArgs};

pub(super) fn quantize_operand(
    module: &Nvfp4QuantModule,
    stream: &CudaStream,
    x: &DeviceBuffer<f32>,
    scratch: &mut Nvfp4TcMatmulOperand<'_>,
    row_count: u32,
    row_len: u32,
    (sign_seed, scale_seed): (u32, u32),
) -> Result<(), DriverError> {
    module.fp32_to_nvfp4_quartet_backward_ms_eden_with_global_scale(
        QuartetBackwardMsEdenQuantArgs {
            stream,
            x,
            out_fp4: scratch.bytes,
            out_scales: scratch.scales,
            out_global_scales: scratch.global_scales,
            out_chunk_amax: scratch.chunk_amax,
            row_count,
            src_row_len: row_len,
            dst_row_len: row_len,
            sign_seed,
            scale_seed,
        },
        scratch.global_scale,
    )?;
    Ok(())
}
