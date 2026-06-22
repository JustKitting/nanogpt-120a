use cuda_core::DeviceBuffer;
use cuda_core::DriverError;
use rust_kernels_cuda::layer_norm::{GptLayerNormArgs, GptLayerNormSaveResidualF16Args};

use super::args::{LayerNormForwardArgs, LayerNormTensors};
use crate::types::{HiddenStateDevice, HiddenVectorShape, LayerNormTensor, Nvfp4ShapeInit};
use crate::{GPT2_LAYER_NORM_EPSILON, GPT2_N_EMBD};

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
            batch_size,
            seq_len,
            row_count,
            residual,
            normalized,
            normalized_amax,
            mean,
            inv_std,
        } = args.hidden;

        args.module.gpt_layer_norm(GptLayerNormArgs {
            stream,
            residual,
            weight: args.tensors.weight,
            bias: args.tensors.bias,
            normalized,
            normalized_amax,
            mean,
            inv_std,
            row_count,
            embedding_dim: GPT2_N_EMBD as u32,
            epsilon: GPT2_LAYER_NORM_EPSILON,
        })?;

        Ok(HiddenStateDevice {
            stream,
            batch_size,
            seq_len,
            row_count,
            residual,
            normalized,
            normalized_amax,
            mean,
            inv_std,
        })
    }

    pub fn forward_save_residual_f16<'a>(
        &self,
        args: LayerNormForwardArgs<'a>,
        residual_f16: &mut DeviceBuffer<u16>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let HiddenStateDevice {
            stream,
            batch_size,
            seq_len,
            row_count,
            residual,
            normalized,
            normalized_amax,
            mean,
            inv_std,
        } = args.hidden;

        args.module
            .gpt_layer_norm_save_residual_f16(GptLayerNormSaveResidualF16Args {
                stream,
                residual,
                weight: args.tensors.weight,
                bias: args.tensors.bias,
                normalized,
                normalized_amax,
                mean,
                inv_std,
                residual_f16,
                row_count,
                embedding_dim: GPT2_N_EMBD as u32,
                epsilon: GPT2_LAYER_NORM_EPSILON,
            })?;

        Ok(HiddenStateDevice {
            stream,
            batch_size,
            seq_len,
            row_count,
            residual,
            normalized,
            normalized_amax,
            mean,
            inv_std,
        })
    }
}
