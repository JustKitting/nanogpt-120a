use cuda_core::{CudaStream, DeviceBuffer, DriverError, memory};
use gpt2_nvfp4::{BlockBackwardGrads, Gpt2BackwardGrads, LayerNormGrads};

use super::grads::BackwardParts;

pub fn clear_backward_parts(
    stream: &CudaStream,
    parts: &mut BackwardParts<'_>,
) -> Result<(), DriverError> {
    clear_buffer(stream, parts.losses)?;
    clear_buffer(stream, parts.d_lm_head_weight)?;
    clear_gpt2_grads(stream, &mut parts.grads)
}

fn clear_gpt2_grads(
    stream: &CudaStream,
    grads: &mut Gpt2BackwardGrads<'_>,
) -> Result<(), DriverError> {
    clear_buffer(stream, grads.dlogits)?;
    clear_buffer(stream, grads.d_embedding_residual)?;
    for block in &mut grads.blocks {
        clear_block_grads(stream, block)?;
    }
    clear_layer_norm_grads(stream, &mut grads.final_norm)
}

fn clear_block_grads(
    stream: &CudaStream,
    grads: &mut BlockBackwardGrads<'_>,
) -> Result<(), DriverError> {
    clear_buffer(stream, grads.d_residual_in)?;
    clear_layer_norm_grads(stream, &mut grads.ln_1)?;
    clear_buffer(stream, grads.d_qkv)?;
    clear_buffer(stream, grads.d_attention_out)?;
    clear_buffer(stream, grads.d_residual_after_attention)?;
    clear_layer_norm_grads(stream, &mut grads.ln_2)?;
    clear_buffer(stream, grads.d_mlp_up)?;
    clear_buffer(stream, grads.d_mlp_relu2)?;
    clear_buffer(stream, grads.d_attn_qkv_weight)?;
    clear_buffer(stream, grads.d_attn_qkv_bias)?;
    clear_buffer(stream, grads.d_attn_c_proj_weight)?;
    clear_buffer(stream, grads.d_attn_c_proj_bias)?;
    clear_buffer(stream, grads.d_mlp_c_fc_weight)?;
    clear_buffer(stream, grads.d_mlp_c_fc_bias)?;
    clear_buffer(stream, grads.d_mlp_c_proj_weight)?;
    clear_buffer(stream, grads.d_mlp_c_proj_bias)?;
    clear_buffer(stream, grads.d_residual_out)
}

fn clear_layer_norm_grads(
    stream: &CudaStream,
    grads: &mut LayerNormGrads<'_>,
) -> Result<(), DriverError> {
    clear_buffer(stream, grads.d_residual)?;
    clear_buffer(stream, grads.d_normalized)?;
    clear_buffer(stream, grads.d_weight)?;
    clear_buffer(stream, grads.d_bias)
}

fn clear_buffer<T>(stream: &CudaStream, buffer: &mut DeviceBuffer<T>) -> Result<(), DriverError> {
    let bytes = buffer.num_bytes();
    if bytes == 0 {
        return Ok(());
    }

    unsafe { memory::memset_d8_async(buffer.cu_deviceptr(), 0, bytes, stream.cu_stream()) }
}
