use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use cuda_core::CudaStream;

mod tensor;
mod topology;

use super::{format::CheckpointWriter, schema};
use crate::upload::UploadedModel;
use crate::{AppResult, fs_utils::ensure_parent};

pub fn save_uploaded_model(stream: &CudaStream, model: &UploadedModel, path: &Path) -> AppResult {
    ensure_parent(path)?;
    let file = File::create(path)?;
    let mut writer = CheckpointWriter::new(BufWriter::new(file));
    writer.write_header(schema::tensor_count(model.blocks.len()))?;
    topology::write_model(&mut writer, stream, model)?;
    writer.finish()
}
