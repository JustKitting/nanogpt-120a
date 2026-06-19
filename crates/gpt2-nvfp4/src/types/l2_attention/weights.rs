use cuda_core::DriverError;

use super::forward;
use super::tape::AttentionForwardTape;
use super::tensors::{AttentionForwardArgs, AttentionProjectionTensors};
use crate::random::InitRng;
use crate::types::{HiddenStateDevice, HiddenStateNvfp4, QkvLinear, ResidualLinear};

#[derive(Clone, Debug)]
pub struct AttentionWeights {
    pub c_attn: QkvLinear,
    pub c_proj: ResidualLinear,
}

impl AttentionWeights {
    pub(crate) fn init(rng: &mut InitRng, residual_projection_scale: f32) -> Self {
        Self {
            c_attn: QkvLinear::init(rng),
            c_proj: ResidualLinear::init_with_weight_scale(rng, residual_projection_scale),
        }
    }

    pub fn input_from_embeddings<'a, 'scratch>(
        module: &'a rust_kernels_cuda::attention::AttentionModule,
        quant_module: &'a rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule,
        input_nvfp4: HiddenStateNvfp4<'scratch>,
        projections: AttentionProjectionTensors<'a>,
        qkv: &'scratch mut cuda_core::DeviceBuffer<f32>,
        attention_log_sum_exp: &'scratch mut cuda_core::DeviceBuffer<f32>,
        hidden: HiddenStateDevice<'a>,
    ) -> AttentionForwardArgs<'a, 'scratch> {
        Self::input_from_embeddings_with_tape(
            module,
            quant_module,
            input_nvfp4,
            projections,
            qkv,
            attention_log_sum_exp,
            hidden,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn input_from_embeddings_with_tape<'a, 'scratch>(
        module: &'a rust_kernels_cuda::attention::AttentionModule,
        quant_module: &'a rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule,
        input_nvfp4: HiddenStateNvfp4<'scratch>,
        projections: AttentionProjectionTensors<'a>,
        qkv: &'scratch mut cuda_core::DeviceBuffer<f32>,
        attention_log_sum_exp: &'scratch mut cuda_core::DeviceBuffer<f32>,
        hidden: HiddenStateDevice<'a>,
        tape: Option<AttentionForwardTape<'scratch>>,
    ) -> AttentionForwardArgs<'a, 'scratch> {
        AttentionForwardArgs {
            module,
            quant_module,
            input_nvfp4,
            projections,
            qkv,
            attention_log_sum_exp,
            hidden,
            tape,
        }
    }

    pub fn forward<'a, 'scratch>(
        args: AttentionForwardArgs<'a, 'scratch>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        forward::forward(args)
    }
}
