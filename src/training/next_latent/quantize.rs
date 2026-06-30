use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantModule, Nvfp4QuantRowwiseArgs, RowAmaxArgs};

use super::buffers::RowwiseQuantizeBuffers;
use super::forward::NextLatForwardArgs;

pub(super) fn quantize_input(args: &mut NextLatForwardArgs<'_, '_>) -> Result<(), DriverError> {
    let row_count = args.row_count;
    let buffers = args.buffers.input_quantize();
    quantize_precomputed_amax(
        args.quant,
        args.stream,
        buffers,
        row_count,
        gpt2_nvfp4::NEXTLAT_INPUT as u32,
    )
}

pub(super) fn quantize_activation(
    quant: &Nvfp4QuantModule,
    stream: &CudaStream,
    row_count: u32,
    buffers: RowwiseQuantizeBuffers<'_>,
) -> Result<(), DriverError> {
    quant.row_amax_f32(RowAmaxArgs {
        stream,
        x: buffers.input,
        out: buffers.amax,
        row_count,
        row_len: gpt2_nvfp4::NEXTLAT_HIDDEN as u32,
    })?;
    quantize_precomputed_amax(
        quant,
        stream,
        buffers,
        row_count,
        gpt2_nvfp4::NEXTLAT_HIDDEN as u32,
    )
}

fn quantize_precomputed_amax(
    quant: &Nvfp4QuantModule,
    stream: &CudaStream,
    buffers: RowwiseQuantizeBuffers<'_>,
    row_count: u32,
    row_len: u32,
) -> Result<(), DriverError> {
    quant.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
        stream,
        x: buffers.input,
        amax: &*buffers.amax,
        out_fp4: buffers.out.bytes,
        out_scales: buffers.out.scales,
        out_global_scale: buffers.out.global_scales,
        group_count: row_count * row_len / 16,
        row_len,
    })
}
