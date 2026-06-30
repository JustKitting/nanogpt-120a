use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::kernels;
use crate::launch::linear_config;

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
        linear_config(
            rows.saturating_mul(dst_cols),
            kernels::PAD_THREADS_PER_BLOCK,
        ),
        src,
        dst,
        rows,
        src_cols,
        dst_cols,
    )
}
