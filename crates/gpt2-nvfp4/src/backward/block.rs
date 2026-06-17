use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::layer_norm_backward::LayerNormBackwardModule;
use rust_kernels_cuda::residual::{ResidualBackwardModule, ResidualGradAddArgs};

use super::layer_norm::{Gpt2LayerNormBackwardArgs, layer_norm_backward};
use super::mlp::{
    MlpBackwardArgs, MlpBackwardGrads, MlpBackwardModules, MlpBackwardScratch, MlpBackwardSeeds,
    backward as mlp_backward,
};
use crate::types::{BlockBackwardGrads, BlockForwardSaved, LayerNormGrads};
use crate::{GPT2_CONTEXT_LEN, GPT2_N_EMBD, LayerNormTensors, MlpProjectionTensors};

#[derive(Clone, Copy)]
pub struct BlockMlpBackwardModules<'a> {
    pub residual: &'a ResidualBackwardModule,
    pub layer_norm: &'a LayerNormBackwardModule,
    pub mlp: MlpBackwardModules<'a>,
}

pub struct BlockMlpBackwardArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub modules: BlockMlpBackwardModules<'a>,
    pub saved: BlockForwardSaved<'a>,
    pub ln_2: LayerNormTensors<'a>,
    pub mlp_projections: MlpProjectionTensors<'a>,
    pub grads: BlockBackwardGrads<'out>,
    pub scratch: MlpBackwardScratch<'scratch>,
    pub seeds: MlpBackwardSeeds,
}

pub fn mlp_side_backward(args: BlockMlpBackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    let BlockMlpBackwardArgs {
        stream,
        modules,
        saved,
        ln_2,
        mlp_projections,
        grads,
        scratch,
        seeds,
    } = args;
    let BlockBackwardGrads {
        d_residual_after_attention,
        ln_2: ln_2_grads,
        d_mlp_up,
        d_mlp_relu2,
        d_mlp_c_fc_weight,
        d_mlp_c_fc_bias,
        d_mlp_c_proj_weight,
        d_mlp_c_proj_bias,
        d_residual_out,
        ..
    } = grads;
    let LayerNormGrads {
        d_residual: d_ln_2_residual,
        d_normalized: d_ln_2_normalized,
        d_weight: d_ln_2_weight,
        d_bias: d_ln_2_bias,
    } = ln_2_grads;

    mlp_backward(MlpBackwardArgs {
        stream,
        modules: modules.mlp,
        saved,
        projections: mlp_projections,
        d_residual_out: &*d_residual_out,
        grads: MlpBackwardGrads {
            d_mlp_relu2,
            d_mlp_up,
            d_ln_2_normalized: &mut *d_ln_2_normalized,
            d_c_proj_weight: d_mlp_c_proj_weight,
            d_c_proj_bias: d_mlp_c_proj_bias,
            d_c_fc_weight: d_mlp_c_fc_weight,
            d_c_fc_bias: d_mlp_c_fc_bias,
        },
        scratch,
        seeds,
    })?;

    layer_norm_backward(Gpt2LayerNormBackwardArgs {
        stream,
        module: modules.layer_norm,
        weights: ln_2,
        saved: saved.ln_2,
        grads: LayerNormGrads {
            d_residual: d_ln_2_residual,
            d_normalized: d_ln_2_normalized,
            d_weight: d_ln_2_weight,
            d_bias: d_ln_2_bias,
        },
    })?;

    modules.residual.grad_add(ResidualGradAddArgs {
        stream,
        direct: &*d_residual_out,
        branch: &*d_ln_2_residual,
        out: d_residual_after_attention,
        len: (GPT2_CONTEXT_LEN * GPT2_N_EMBD) as u32,
    })
}
