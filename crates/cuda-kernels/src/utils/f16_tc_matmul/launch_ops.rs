use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use super::kernels::{F16_THREADS_PER_BLOCK, LoadedModule};

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
        linear_config(rows * dst_cols),
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
        linear_config(element_count.div_ceil(2)),
        src,
        dst,
        element_count,
    )
}

fn linear_config(element_count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (element_count.div_ceil(F16_THREADS_PER_BLOCK), 1, 1),
        block_dim: (F16_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
