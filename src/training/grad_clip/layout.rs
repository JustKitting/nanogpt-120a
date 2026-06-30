use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD, GPT2_QKV, GPT2_VOCAB_SIZE, NEXTLAT_HIDDEN, NEXTLAT_INPUT};
use rust_kernels_cuda::optimizer::GRAD_CLIP_VALUES_PER_CHUNK;

use crate::training::grad_block::LayerNormGradBuffers;
use crate::training::grads::BackwardBuffers;
use crate::training::next_latent::NextLatGradBuffers;

#[derive(Clone, Copy)]
pub(super) struct HostGradPtr {
    pub(super) ptr: u64,
    pub(super) len: u32,
    pub(super) chunk_offset: u32,
}

struct HostGradView<'a> {
    name: String,
    buffer: &'a DeviceBuffer<f32>,
    len: usize,
}

pub(in crate::training) struct NonFiniteGradient {
    pub(in crate::training) name: String,
    pub(in crate::training) index: usize,
    pub(in crate::training) value: f32,
}

pub(super) fn parameter_gradients(
    grads: &BackwardBuffers,
    next_latent: &NextLatGradBuffers,
) -> Vec<HostGradPtr> {
    let views = parameter_gradient_views(grads, next_latent);
    let mut rows = Vec::new();
    for view in views {
        push(&mut rows, view.buffer, view.len);
    }
    rows
}

pub(super) fn gradient_chunk_count(rows: &[HostGradPtr]) -> u32 {
    rows.last()
        .map(|row| row.chunk_offset + chunks(row.len))
        .unwrap_or(0)
}

pub(in crate::training) fn first_non_finite_gradient(
    stream: &CudaStream,
    grads: &BackwardBuffers,
    next_latent: &NextLatGradBuffers,
) -> Result<Option<NonFiniteGradient>, DriverError> {
    first_non_finite_view(stream, parameter_gradient_views(grads, next_latent))
}

fn first_non_finite_view<'a>(
    stream: &CudaStream,
    views: Vec<HostGradView<'a>>,
) -> Result<Option<NonFiniteGradient>, DriverError> {
    for view in views {
        let values = view.buffer.to_host_vec(stream)?;
        if let Some((index, value)) = values
            .iter()
            .take(view.len)
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Ok(Some(NonFiniteGradient {
                name: view.name,
                index,
                value,
            }));
        }
    }
    Ok(None)
}

fn parameter_gradient_views<'a>(
    grads: &'a BackwardBuffers,
    next_latent: &'a NextLatGradBuffers,
) -> Vec<HostGradView<'a>> {
    let mut rows = Vec::new();
    push_view(
        &mut rows,
        "lm_head.weight",
        &grads.d_lm_head_weight,
        GPT2_VOCAB_SIZE * GPT2_N_EMBD,
    );
    push_layer_norm_views(&mut rows, "final_norm", &grads.final_norm);

    for (block_index, block) in grads.blocks.iter().enumerate() {
        let prefix = format!("blocks.{block_index}");
        push_layer_norm_views(&mut rows, &format!("{prefix}.ln_1"), &block.ln_1);
        push_prefixed_views(
            &mut rows,
            &prefix,
            &[
                (
                    "attn_qkv.weight",
                    &block.d_attn_qkv_weight,
                    GPT2_N_EMBD * GPT2_QKV,
                ),
                ("attn_qkv.bias", &block.d_attn_qkv_bias, GPT2_QKV),
                (
                    "attn_c_proj.weight",
                    &block.d_attn_c_proj_weight,
                    GPT2_N_EMBD * GPT2_N_EMBD,
                ),
                ("attn_c_proj.bias", &block.d_attn_c_proj_bias, GPT2_N_EMBD),
            ],
        );
        push_layer_norm_views(&mut rows, &format!("{prefix}.ln_2"), &block.ln_2);
        push_prefixed_views(
            &mut rows,
            &prefix,
            &[
                (
                    "mlp_up.weight",
                    &block.d_mlp_c_fc_weight,
                    GPT2_N_EMBD * GPT2_MLP,
                ),
                ("mlp_up.bias", &block.d_mlp_c_fc_bias, GPT2_MLP),
                (
                    "mlp_down.weight",
                    &block.d_mlp_c_proj_weight,
                    GPT2_MLP * GPT2_N_EMBD,
                ),
                ("mlp_down.bias", &block.d_mlp_c_proj_bias, GPT2_N_EMBD),
            ],
        );
    }
    push_prefixed_views(
        &mut rows,
        "next_latent",
        &[
            ("norm.weight", &next_latent.d_norm_weight, NEXTLAT_INPUT),
            ("norm.bias", &next_latent.d_norm_bias, NEXTLAT_INPUT),
            (
                "input_projection.weight",
                &next_latent.d_input_projection_weight,
                NEXTLAT_INPUT * NEXTLAT_HIDDEN,
            ),
            (
                "input_projection.bias",
                &next_latent.d_input_projection_bias,
                NEXTLAT_HIDDEN,
            ),
            (
                "transition.weight",
                &next_latent.d_transition_weight,
                NEXTLAT_HIDDEN * NEXTLAT_HIDDEN,
            ),
            (
                "transition.bias",
                &next_latent.d_transition_bias,
                NEXTLAT_HIDDEN,
            ),
            (
                "output_projection.weight",
                &next_latent.d_output_projection_weight,
                NEXTLAT_HIDDEN * GPT2_N_EMBD,
            ),
            (
                "output_projection.bias",
                &next_latent.d_output_projection_bias,
                GPT2_N_EMBD,
            ),
        ],
    );

    rows
}

fn push_layer_norm_views<'a>(
    rows: &mut Vec<HostGradView<'a>>,
    name: &str,
    grads: &'a LayerNormGradBuffers,
) {
    push_prefixed_views(
        rows,
        name,
        &[
            ("weight", &grads.d_weight, GPT2_N_EMBD),
            ("bias", &grads.d_bias, GPT2_N_EMBD),
        ],
    );
}

fn push_prefixed_views<'a>(
    rows: &mut Vec<HostGradView<'a>>,
    prefix: &str,
    views: &[(&str, &'a DeviceBuffer<f32>, usize)],
) {
    for &(name, buffer, len) in views {
        push_view(rows, &format!("{prefix}.{name}"), buffer, len);
    }
}

fn push_view<'a>(
    rows: &mut Vec<HostGradView<'a>>,
    name: &str,
    buffer: &'a DeviceBuffer<f32>,
    len: usize,
) {
    rows.push(HostGradView::new(name, buffer, len));
}

impl<'a> HostGradView<'a> {
    fn new(name: &str, buffer: &'a DeviceBuffer<f32>, len: usize) -> Self {
        Self {
            name: name.to_string(),
            buffer,
            len,
        }
    }
}

fn push(rows: &mut Vec<HostGradPtr>, buffer: &DeviceBuffer<f32>, len: usize) {
    let chunk_offset = gradient_chunk_count(rows);
    rows.push(HostGradPtr {
        ptr: buffer.cu_deviceptr(),
        len: len as u32,
        chunk_offset,
    });
}

fn chunks(len: u32) -> u32 {
    len.div_ceil(GRAD_CLIP_VALUES_PER_CHUNK as u32)
}
