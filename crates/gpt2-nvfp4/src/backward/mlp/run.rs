use cuda_core::DriverError;
use rust_kernels_cuda::mlp::Relu2BackwardArgs;

use super::args::{MlpBackwardArgs, MlpBackwardGrads, MlpBackwardScratch};
use super::pass::{LinearPass, run_linear_pass};
use crate::{GPT2_MLP, GPT2_N_EMBD};

pub fn backward(args: MlpBackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    let MlpBackwardArgs {
        stream,
        modules,
        saved,
        projections,
        d_residual_out,
        grads,
        scratch,
        seeds,
    } = args;
    let MlpBackwardScratch {
        down_linear,
        up_linear,
        ..
    } = scratch;
    let MlpBackwardGrads {
        d_mlp_relu2,
        d_mlp_up,
        d_ln_2_normalized,
        d_c_proj_weight,
        d_c_proj_bias,
        d_c_fc_weight,
        d_c_fc_bias,
    } = grads;

    run_linear_pass(
        &modules,
        stream,
        LinearPass {
            e: d_residual_out,
            saved_input: saved.mlp_down_input_nvfp4,
            weight: projections.down.weight,
            linear_scratch: down_linear,
            dinput: d_mlp_relu2,
            dweight: d_c_proj_weight,
            dbias: d_c_proj_bias,
            row_count: saved.row_count,
            input_dim: GPT2_MLP as u32,
            output_dim: GPT2_N_EMBD as u32,
            sign_seed: seeds.down_sign,
            scale_seed: seeds.down_scale,
        },
    )?;

    modules.mlp.relu2_backward(Relu2BackwardArgs {
        stream,
        pre_activation: saved.mlp_up,
        d_out: d_mlp_relu2,
        d_pre_activation: d_mlp_up,
        len: saved.row_count * GPT2_MLP as u32,
    })?;

    run_linear_pass(
        &modules,
        stream,
        LinearPass {
            e: d_mlp_up,
            saved_input: saved.mlp_up_input_nvfp4,
            weight: projections.up.weight,
            linear_scratch: up_linear,
            dinput: d_ln_2_normalized,
            dweight: d_c_fc_weight,
            dbias: d_c_fc_bias,
            row_count: saved.row_count,
            input_dim: GPT2_N_EMBD as u32,
            output_dim: GPT2_MLP as u32,
            sign_seed: seeds.up_sign,
            scale_seed: seeds.up_scale,
        },
    )
}
