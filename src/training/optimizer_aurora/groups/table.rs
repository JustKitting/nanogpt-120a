use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::{AuroraGroupTable, HostPtrs};

pub(super) fn upload_table(
    stream: &CudaStream,
    rows: &[HostPtrs],
) -> Result<AuroraGroupTable, DriverError> {
    let values: Vec<_> = rows.iter().copied().map(HostPtrs::descriptor).collect();
    Ok(AuroraGroupTable {
        slots: DeviceBuffer::from_host(stream, &values)?,
        host_slots: values,
    })
}
