use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use super::kernels;

pub(super) fn pad_rows(
    module: &kernels::module::LoadedModule,
    stream: &CudaStream,
    src: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
    rows: u32,
    src_cols: u32,
    dst_cols: u32,
) -> Result<(), DriverError> {
    module.fp32_pad_rows_kernel(
        stream,
        LaunchConfig {
            grid_dim: (
                rows.saturating_mul(dst_cols)
                    .div_ceil(kernels::PAD_THREADS_PER_BLOCK),
                1,
                1,
            ),
            block_dim: (kernels::PAD_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        },
        src,
        dst,
        rows,
        src_cols,
        dst_cols,
    )
}
