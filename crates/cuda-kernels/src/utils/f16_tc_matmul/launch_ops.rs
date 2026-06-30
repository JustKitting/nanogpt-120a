use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::kernels::{F16_THREADS_PER_BLOCK, LoadedModule};
use crate::launch::linear_config;

pub(super) fn pad_rows(
    module: &LoadedModule,
    stream: &CudaStream,
    src: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
    rows: u32,
    src_cols: u32,
    dst_cols: u32,
) -> Result<(), DriverError> {
    module.f16_fp32_pad_rows_kernel(
        stream,
        linear_config(rows * dst_cols, F16_THREADS_PER_BLOCK),
        src,
        dst,
        rows,
        src_cols,
        dst_cols,
    )
}

pub(super) fn convert(
    module: &LoadedModule,
    stream: &CudaStream,
    src: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<u16>,
    element_count: u32,
) -> Result<(), DriverError> {
    module.fp32_to_f16_kernel(
        stream,
        linear_config(element_count.div_ceil(2), F16_THREADS_PER_BLOCK),
        src,
        dst,
        element_count,
    )
}
