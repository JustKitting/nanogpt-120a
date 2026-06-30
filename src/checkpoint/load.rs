use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::GPT2_N_LAYER;

use super::format::{CheckpointReader, CheckpointTensor};
use crate::AppResult;
use crate::upload::{
    UploadedBlock, UploadedLayerNorm, UploadedLinear, UploadedModel, UploadedNextLat, UploadedNvfp4,
};

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
