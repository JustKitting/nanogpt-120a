use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD, GPT2_QKV, GPT2_VOCAB_SIZE, NEXTLAT_HIDDEN, NEXTLAT_INPUT};

use crate::training::{
    grad_block::LayerNormGradBuffers, grads::BackwardBuffers, next_latent::NextLatGradBuffers,
};

pub(super) struct HostGradView<'a> {
    pub(super) name: String,
    pub(super) buffer: &'a DeviceBuffer<f32>,
    pub(super) len: usize,
}

pub(super) fn parameter_gradient_views<'a>(
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
