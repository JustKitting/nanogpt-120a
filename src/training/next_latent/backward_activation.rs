use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::next_latent::{NextLatGeluBackwardArgs, NextLatModule};

pub(super) fn gelu_backward(
    module: &NextLatModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    d_out: &DeviceBuffer<f32>,
    d_input: &mut DeviceBuffer<f32>,
    len: u32,
) -> Result<(), DriverError> {
    module.gelu_backward(NextLatGeluBackwardArgs {
        stream,
        input,
        d_out,
        d_input,
        len,
    })
}
