use cuda_core::CudaStream;

use super::super::format::CheckpointWriter;
use crate::{AppResult, upload::UploadedNvfp4};

pub(super) fn write(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    name: &str,
    tensor: &UploadedNvfp4,
) -> AppResult {
    writer.write_tensor(
        name,
        tensor.len,
        tensor.global_scale.to_host_vec(stream)?[0],
        &tensor.bytes.to_host_vec(stream)?,
        &tensor.scales.to_host_vec(stream)?,
    )
}
