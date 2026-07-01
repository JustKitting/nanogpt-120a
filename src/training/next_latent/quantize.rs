use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use super::buffers::RowwiseQuantizeBuffers;
use super::forward::NextLatForwardArgs;

pub(super) fn quantize_input(args: &mut NextLatForwardArgs<'_, '_>) -> Result<(), DriverError> {
    let row_count = args.row_count;
    let mut buffers = args.buffers.input_quantize();
    buffers.out.quantize_precomputed_amax(
        args.quant,
        args.stream,
        buffers.input,
        buffers.amax,
        row_count,
        gpt2_nvfp4::NEXTLAT_INPUT_DIM,
    )
}

pub(super) fn quantize_activation(
    quant: &Nvfp4QuantModule,
    stream: &CudaStream,
    row_count: u32,
    mut buffers: RowwiseQuantizeBuffers<'_>,
) -> Result<(), DriverError> {
    buffers.out.quantize_row_amax(
        quant,
        stream,
        buffers.input,
        buffers.amax,
        row_count,
        gpt2_nvfp4::NEXTLAT_HIDDEN_DIM,
    )
}
