use cuda_core::DriverError;
use rust_kernels_cuda::layer_norm::{GptLayerNormArgs, LayerNormModule};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

use crate::{GPT2_CONTEXT_LEN, GPT2_LAYER_NORM_EPSILON, GPT2_N_EMBD};

use super::{HiddenStateDevice, HiddenVectorShape, LayerNormTensor, Nvfp4ShapeInit};

#[derive(Clone, Copy)]
pub struct LayerNormTensors<'a> {
    pub weight: Nvfp4DeviceTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
}

pub struct LayerNormForwardArgs<'a> {
    pub module: &'a LayerNormModule,
    pub tensors: LayerNormTensors<'a>,
    pub hidden: HiddenStateDevice<'a>,
}

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
        module: &'a LayerNormModule,
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
