use std::collections::HashMap;

use cuda_core::CudaStream;

use super::super::format::CheckpointTensor;
use crate::{AppResult, upload::UploadedNvfp4};

pub(super) fn take_uploaded(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
    name: &str,
) -> AppResult<UploadedNvfp4> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| format!("checkpoint is missing tensor {name}"))?;
    validate(name, &tensor)?;
    UploadedNvfp4::from_host(
        stream,
        &tensor.bytes,
        &tensor.scales,
        tensor.global_scale,
        tensor.len,
    )
}

fn validate(name: &str, tensor: &CheckpointTensor) -> AppResult {
    validate_len(name, "bytes", tensor.bytes.len(), tensor.len / 2)?;
    validate_len(name, "scales", tensor.scales.len(), tensor.len / 16)
}

fn validate_len(name: &str, field: &str, actual: usize, expected: usize) -> AppResult {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{name} has {actual} {field}; expected {expected}").into())
    }
}
