use cuda_core::{DeviceBuffer, DriverError};

use super::types::{Gpt2BackwardModules, Gpt2BackwardWeights};
use crate::backward::{
    FinalHeadBackwardArgs, FinalHeadBackwardScratch, FinalHeadBackwardSeeds, final_head_backward,
};
use crate::types::Gpt2ForwardSaved;

#[allow(clippy::too_many_arguments)]
pub(super) fn run_final_head<'a, 'scratch, 'out>(
    stream: &'a cuda_core::CudaStream,
    modules: Gpt2BackwardModules<'a>,
    saved: Gpt2ForwardSaved<'a>,
    weights: Gpt2BackwardWeights<'a>,
    targets: &'a DeviceBuffer<u32>,
    losses: &'out mut DeviceBuffer<f32>,
    dlogits: &'out mut DeviceBuffer<f32>,
    d_final_normalized: &'out mut DeviceBuffer<f32>,
    d_lm_head_weight: &'out mut DeviceBuffer<f32>,
    scratch: FinalHeadBackwardScratch<'scratch>,
    seeds: FinalHeadBackwardSeeds,
) -> Result<(), DriverError> {
    final_head_backward(FinalHeadBackwardArgs {
        stream,
        modules: modules.final_head,
        logits: saved.logits,
        targets,
        final_normalized: saved.lm_head_input_nvfp4,
        lm_head_weight: weights.lm_head_weight,
        losses,
        dlogits,
        d_final_normalized,
        d_lm_head_weight,
        scratch,
        seeds,
    })
}
