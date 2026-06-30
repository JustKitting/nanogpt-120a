use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::layer_norm::{
    GptLayerNormArgs, GptLayerNormSaveResidualF16Args, LayerNormModule,
};

use super::args::{LayerNormForwardArgs, LayerNormTensors};
use super::weights::LayerNormWeights;
use crate::types::HiddenStateDevice;
use crate::{GPT2_LAYER_NORM_EPSILON, GPT2_N_EMBD};

impl LayerNormWeights {
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
        let hidden = args.hidden;

        args.module.gpt_layer_norm(GptLayerNormArgs {
            stream: hidden.stream,
            residual: &mut *hidden.residual,
            weight: args.tensors.weight,
            bias: args.tensors.bias,
            normalized: &mut *hidden.normalized,
            normalized_amax: &mut *hidden.normalized_amax,
            mean: &mut *hidden.mean,
            inv_std: &mut *hidden.inv_std,
            row_count: hidden.row_count,
            embedding_dim: GPT2_N_EMBD as u32,
            epsilon: GPT2_LAYER_NORM_EPSILON,
        })?;

        Ok(hidden)
    }

    pub fn forward_save_residual_f16<'a>(
        &self,
        args: LayerNormForwardArgs<'a>,
        residual_f16: &mut DeviceBuffer<u16>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let hidden = args.hidden;

        args.module
            .gpt_layer_norm_save_residual_f16(GptLayerNormSaveResidualF16Args {
                stream: hidden.stream,
                residual: &mut *hidden.residual,
                weight: args.tensors.weight,
                bias: args.tensors.bias,
                normalized: &mut *hidden.normalized,
                normalized_amax: &mut *hidden.normalized_amax,
                mean: &mut *hidden.mean,
                inv_std: &mut *hidden.inv_std,
                residual_f16,
                row_count: hidden.row_count,
                embedding_dim: GPT2_N_EMBD as u32,
                epsilon: GPT2_LAYER_NORM_EPSILON,
            })?;

        Ok(hidden)
    }
}
