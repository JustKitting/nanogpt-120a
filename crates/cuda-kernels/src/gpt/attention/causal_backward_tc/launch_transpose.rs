use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::kernels;
use super::types::CausalAttentionBackwardTcScratch;

pub(super) struct TransposeShape {
    pub batch_head: u32,
    pub seq_len: u32,
}

pub(super) fn run_transposes(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    scratch: &mut CausalAttentionBackwardTcScratch<'_>,
    shape: TransposeShape,
) -> Result<(), DriverError> {
    transpose(
        module,
        stream,
        scratch.p,
        scratch.p_t,
        shape.batch_head,
        shape.seq_len,
        shape.seq_len,
    )?;
    transpose(
        module,
        stream,
        scratch.ds,
        scratch.ds_t,
        shape.batch_head,
        shape.seq_len,
        shape.seq_len,
    )
}

fn transpose(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    src: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
    batch_count: u32,
    rows: u32,
    cols: u32,
) -> Result<(), DriverError> {
    module.transpose_matrix_kernel(
        stream,
        super::launch_config::linear_config(batch_count * rows * cols),
        src,
        dst,
        batch_count,
        rows,
        cols,
    )
}
