use gpt2_nvfp4::{GPT2_N_EMBD, NEXTLAT_HIDDEN, NEXTLAT_INPUT};

use crate::training::next_latent::NextLatGradBuffers;

use super::{HostGradView, push_prefixed_views};

pub(super) fn push_next_latent_views<'a>(
    rows: &mut Vec<HostGradView<'a>>,
    next_latent: &'a NextLatGradBuffers,
) {
    push_prefixed_views(
        rows,
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
}
