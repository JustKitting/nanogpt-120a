use cuda_core::DriverError;
use rust_kernels_cuda::layer_norm::GptLayerNormArgs;

use super::args::{LayerNormForwardArgs, LayerNormTensors};
use crate::types::{HiddenStateDevice, HiddenVectorShape, LayerNormTensor, Nvfp4ShapeInit};
use crate::{GPT2_CONTEXT_LEN, GPT2_LAYER_NORM_EPSILON, GPT2_N_EMBD};

#[derive(Clone, Debug)]
pub struct LayerNormWeights {
    pub weight: LayerNormTensor,
    pub bias: LayerNormTensor,
}

impl LayerNormWeights {
    pub(crate) fn init() -> Self {
        Self {
            weight: HiddenVectorShape::one_tensor(),
            bias: HiddenVectorShape::zero_tensor(),
        }
    }

    pub fn input_from_block<'a>(
        module: &'a rust_kernels_cuda::layer_norm::LayerNormModule,
        tensors: LayerNormTensors<'a>,
        hidden: HiddenStateDevice<'a>,
    ) -> LayerNormForwardArgs<'a> {
        LayerNormForwardArgs {
            module,
            tensors,
            hidden,
        }
    }

    pub fn forward<'a>(
        &self,
        args: LayerNormForwardArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let HiddenStateDevice {
            stream,
            residual,
            normalized,
            normalized_amax,
        } = args.hidden;

        args.module.gpt_layer_norm(GptLayerNormArgs {
            stream,
            residual,
            weight: args.tensors.weight,
            bias: args.tensors.bias,
            normalized,
            normalized_amax,
            row_count: GPT2_CONTEXT_LEN as u32,
            embedding_dim: GPT2_N_EMBD as u32,
            epsilon: GPT2_LAYER_NORM_EPSILON,
        })?;

        Ok(HiddenStateDevice {
            stream,
            residual,
            normalized,
            normalized_amax,
        })
    }
}
