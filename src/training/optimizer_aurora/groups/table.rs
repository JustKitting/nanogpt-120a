use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError};

use super::{AuroraGroupTable, HostPtrs};

pub(super) fn upload_table(
    stream: &CudaStream,
    rows: &[HostPtrs],
) -> Result<AuroraGroupTable, DriverError> {
    Ok(AuroraGroupTable {
        grad: upload(stream, rows, |p| p.grad)?,
        momentum: upload(stream, rows, |p| p.momentum)?,
        z_master: upload(stream, rows, |p| p.z_master)?,
        x_master: upload(stream, rows, |p| p.x_master)?,
        bytes: upload(stream, rows, |p| p.bytes)?,
        scales: upload(stream, rows, |p| p.scales)?,
        global_scale: upload(stream, rows, |p| p.global_scale)?,
        rows: upload(stream, rows, |p| p.rows)?,
        cols: upload(stream, rows, |p| p.cols)?,
        learning_rate_multipliers: upload(stream, rows, |p| p.learning_rate_multiplier)?,
    })
}

fn upload<T, F>(
    stream: &CudaStream,
    rows: &[HostPtrs],
    f: F,
) -> Result<DeviceBuffer<T>, DriverError>
where
    T: DeviceCopy,
    F: Fn(HostPtrs) -> T,
{
    let values: Vec<T> = rows.iter().copied().map(f).collect();
    DeviceBuffer::from_host(stream, &values)
}
