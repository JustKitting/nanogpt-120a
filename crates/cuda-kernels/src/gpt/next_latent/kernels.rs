use cuda_device::{DisjointSlice, cuda_module, kernel};

use super::args::NextLatShape;

#[path = "kernels/body.rs"]
mod body;
use body::{nextlat_concat_backward_body, nextlat_concat_input_body, nextlat_smooth_l1_body};

#[cuda_module]
pub mod module {
    use super::*;

    #[kernel]
    pub fn nextlat_concat_input_kernel(
        next_token_embeddings: &[f32],
        current_states: &[f32],
        mut out: DisjointSlice<f32>,
        shape: NextLatShape,
    ) {
        nextlat_concat_input_body(next_token_embeddings, current_states, &mut out, shape);
    }

    #[kernel]
    pub fn nextlat_concat_backward_kernel(
        d_concat: &[f32],
        d_predicted: &[f32],
        mut d_next_token_embeddings: DisjointSlice<f32>,
        mut d_current_states: DisjointSlice<f32>,
        shape: NextLatShape,
    ) {
        nextlat_concat_backward_body(
            d_concat,
            d_predicted,
            &mut d_next_token_embeddings,
            &mut d_current_states,
            shape,
        );
    }

    #[kernel]
    pub fn nextlat_smooth_l1_kernel(
        predicted_next_states: &[f32],
        target_states: &[f32],
        mut losses: DisjointSlice<f32>,
        mut d_predicted_next_states: DisjointSlice<f32>,
        shape: NextLatShape,
    ) {
        nextlat_smooth_l1_body(
            predicted_next_states,
            target_states,
            &mut losses,
            &mut d_predicted_next_states,
            shape,
        );
    }
}
