mod format;

use std::collections::HashMap;
use std::fs::{File, create_dir_all};
use std::io::{BufReader, BufWriter};
use std::path::Path;

use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::GPT2_N_LAYER;

use crate::AppResult;
use crate::checkpoint::format::{CheckpointReader, CheckpointTensor, CheckpointWriter};
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
    writer.write_header(tensor_count(model))?;
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

pub fn load_uploaded_model(stream: &CudaStream, path: &Path) -> AppResult<UploadedModel> {
    let file = File::open(path)?;
    let mut reader = CheckpointReader::new(BufReader::new(file));
    let tensor_count = reader.read_header()?;
    if tensor_count != expected_tensor_count() {
        return Err(format!(
            "checkpoint has {tensor_count} tensors; expected {}",
            expected_tensor_count()
        )
        .into());
    }

    let mut tensors = HashMap::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let tensor = reader.read_tensor()?;
        if tensors.insert(tensor.name.clone(), tensor).is_some() {
            return Err("checkpoint contains duplicate tensor name".into());
        }
    }

    Ok(UploadedModel {
        token_embedding: take_uploaded_tensor(stream, &mut tensors, "token_embedding")?,
        blocks: load_blocks(stream, &mut tensors)?,
        ln_f: load_layer_norm(stream, &mut tensors, "ln_f")?,
        next_latent: load_next_latent(stream, &mut tensors)?,
    })
}

fn write_next_latent(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    next_latent: &UploadedNextLat,
) -> AppResult {
    write_layer_norm(writer, stream, "next_latent.norm", &next_latent.norm)?;
    write_linear(
        writer,
        stream,
        "next_latent.input_projection",
        &next_latent.input_projection,
    )?;
    write_linear(
        writer,
        stream,
        "next_latent.transition",
        &next_latent.transition,
    )?;
    write_linear(
        writer,
        stream,
        "next_latent.output_projection",
        &next_latent.output_projection,
    )
}

fn write_block(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    index: usize,
    block: &UploadedBlock,
) -> AppResult {
    write_layer_norm(writer, stream, &format!("blocks.{index}.ln_1"), &block.ln_1)?;
    write_linear(
        writer,
        stream,
        &format!("blocks.{index}.attn_qkv"),
        &block.attn_qkv,
    )?;
    write_linear(
        writer,
        stream,
        &format!("blocks.{index}.attn_c_proj"),
        &block.attn_c_proj,
    )?;
    write_layer_norm(writer, stream, &format!("blocks.{index}.ln_2"), &block.ln_2)?;
    write_linear(
        writer,
        stream,
        &format!("blocks.{index}.mlp_up"),
        &block.mlp_up,
    )?;
    write_linear(
        writer,
        stream,
        &format!("blocks.{index}.mlp_down"),
        &block.mlp_down,
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

fn tensor_count(model: &UploadedModel) -> u32 {
    1 + 2 + 8 + model.blocks.len() as u32 * 12
}

fn expected_tensor_count() -> u32 {
    1 + 2 + 8 + GPT2_N_LAYER as u32 * 12
}

fn load_blocks(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
) -> AppResult<Vec<UploadedBlock>> {
    let mut blocks = Vec::with_capacity(GPT2_N_LAYER);
    for index in 0..GPT2_N_LAYER {
        blocks.push(UploadedBlock {
            ln_1: load_layer_norm(stream, tensors, &format!("blocks.{index}.ln_1"))?,
            attn_qkv: load_linear(stream, tensors, &format!("blocks.{index}.attn_qkv"))?,
            attn_c_proj: load_linear(stream, tensors, &format!("blocks.{index}.attn_c_proj"))?,
            ln_2: load_layer_norm(stream, tensors, &format!("blocks.{index}.ln_2"))?,
            mlp_up: load_linear(stream, tensors, &format!("blocks.{index}.mlp_up"))?,
            mlp_down: load_linear(stream, tensors, &format!("blocks.{index}.mlp_down"))?,
        });
    }
    Ok(blocks)
}

fn load_layer_norm(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
    prefix: &str,
) -> AppResult<UploadedLayerNorm> {
    Ok(UploadedLayerNorm {
        weight: take_uploaded_tensor(stream, tensors, &format!("{prefix}.weight"))?,
        bias: take_uploaded_tensor(stream, tensors, &format!("{prefix}.bias"))?,
    })
}

fn load_next_latent(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
) -> AppResult<UploadedNextLat> {
    Ok(UploadedNextLat {
        norm: load_layer_norm(stream, tensors, "next_latent.norm")?,
        input_projection: load_linear(stream, tensors, "next_latent.input_projection")?,
        transition: load_linear(stream, tensors, "next_latent.transition")?,
        output_projection: load_linear(stream, tensors, "next_latent.output_projection")?,
    })
}

fn load_linear(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
    prefix: &str,
) -> AppResult<UploadedLinear> {
    Ok(UploadedLinear {
        weight: take_uploaded_tensor(stream, tensors, &format!("{prefix}.weight"))?,
        bias: take_uploaded_tensor(stream, tensors, &format!("{prefix}.bias"))?,
    })
}

fn take_uploaded_tensor(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
    name: &str,
) -> AppResult<UploadedNvfp4> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| format!("checkpoint is missing tensor {name}"))?;
    validate_tensor(name, &tensor)?;
    Ok(UploadedNvfp4 {
        bytes: DeviceBuffer::from_host(stream, &tensor.bytes)?,
        scales: DeviceBuffer::from_host(stream, &tensor.scales)?,
        global_scale: DeviceBuffer::from_host(stream, &[tensor.global_scale])?,
        len: tensor.len,
    })
}

fn validate_tensor(name: &str, tensor: &CheckpointTensor) -> AppResult {
    if tensor.bytes.len() != tensor.len / 2 {
        return Err(format!(
            "{name} has {} bytes; expected {}",
            tensor.bytes.len(),
            tensor.len / 2
        )
        .into());
    }
    if tensor.scales.len() != tensor.len / 16 {
        return Err(format!(
            "{name} has {} scales; expected {}",
            tensor.scales.len(),
            tensor.len / 16
        )
        .into());
    }
    Ok(())
}
