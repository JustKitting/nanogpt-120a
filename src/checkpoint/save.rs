use std::fs::{File, create_dir_all};
use std::io::BufWriter;
use std::path::Path;

use cuda_core::CudaStream;

use super::{format::CheckpointWriter, schema};
use crate::AppResult;
use crate::upload::{
    UploadedBlock, UploadedLayerNorm, UploadedLinear, UploadedModel, UploadedNextLat, UploadedNvfp4,
};

pub fn save_uploaded_model(stream: &CudaStream, model: &UploadedModel, path: &Path) -> AppResult {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        create_dir_all(parent)?;
    }

    let file = File::create(path)?;
    let mut writer = CheckpointWriter::new(BufWriter::new(file));
    writer.write_header(schema::tensor_count(model.blocks.len()))?;
    write_tensor(
        &mut writer,
        stream,
        "token_embedding",
        &model.token_embedding,
    )?;
    for (index, block) in model.blocks.iter().enumerate() {
        write_block(&mut writer, stream, index, block)?;
    }
    write_layer_norm(&mut writer, stream, "ln_f", &model.ln_f)?;
    write_next_latent(&mut writer, stream, &model.next_latent)?;
    writer.finish()
}

fn write_next_latent(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    next_latent: &UploadedNextLat,
) -> AppResult {
    write_layer_norm(writer, stream, "next_latent.norm", &next_latent.norm)?;
    write_linears(
        writer,
        stream,
        "next_latent",
        [
            ("input_projection", &next_latent.input_projection),
            ("transition", &next_latent.transition),
            ("output_projection", &next_latent.output_projection),
        ],
    )
}

fn write_block(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    index: usize,
    block: &UploadedBlock,
) -> AppResult {
    let prefix = format!("blocks.{index}");
    write_layer_norm(writer, stream, &format!("{prefix}.ln_1"), &block.ln_1)?;
    write_linears(
        writer,
        stream,
        &prefix,
        [
            ("attn_qkv", &block.attn_qkv),
            ("attn_c_proj", &block.attn_c_proj),
        ],
    )?;
    write_layer_norm(writer, stream, &format!("{prefix}.ln_2"), &block.ln_2)?;
    write_linears(
        writer,
        stream,
        &prefix,
        [("mlp_up", &block.mlp_up), ("mlp_down", &block.mlp_down)],
    )
}

fn write_layer_norm(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    prefix: &str,
    layer_norm: &UploadedLayerNorm,
) -> AppResult {
    write_tensor(
        writer,
        stream,
        &format!("{prefix}.weight"),
        &layer_norm.weight,
    )?;
    write_tensor(writer, stream, &format!("{prefix}.bias"), &layer_norm.bias)
}

fn write_linear(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    prefix: &str,
    linear: &UploadedLinear,
) -> AppResult {
    write_tensor(writer, stream, &format!("{prefix}.weight"), &linear.weight)?;
    write_tensor(writer, stream, &format!("{prefix}.bias"), &linear.bias)
}

fn write_linears<const N: usize>(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    prefix: &str,
    linears: [(&str, &UploadedLinear); N],
) -> AppResult {
    for (suffix, linear) in linears {
        write_linear(writer, stream, &format!("{prefix}.{suffix}"), linear)?;
    }
    Ok(())
}

fn write_tensor(
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
