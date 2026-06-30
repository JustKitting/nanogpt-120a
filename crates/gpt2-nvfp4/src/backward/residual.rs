use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::residual::{ResidualBackwardModule, ResidualGradAddArgs};

pub(super) fn residual_grad_add(
    module: &ResidualBackwardModule,
    stream: &CudaStream,
    direct: &DeviceBuffer<f32>,
    branch: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    row_count: u32,
) -> Result<(), DriverError> {
    module.grad_add(ResidualGradAddArgs {
        stream,
        direct,
        branch,
        out,
        len: row_count * crate::GPT2_N_EMBD as u32,
    })
}
