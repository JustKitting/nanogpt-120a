use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;

use super::args::BlockForwardArgs;
use super::weights::Gpt2BlockWeights;
use crate::types::{
    AttentionForwardArgs, AttentionForwardTape, AttentionWeights, HiddenStateDevice,
    LayerNormForwardArgs, LayerNormTape, LayerNormWeights, MlpForwardArgs, MlpScratch, MlpWeights,
};

impl Gpt2BlockWeights {
    pub fn forward<'a, 'scratch>(
        &self,
        args: BlockForwardArgs<'a, 'scratch>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let qkv = args.qkv;
        let attention_log_sum_exp = args.attention_log_sum_exp;
        let mlp_pre_activation = args.mlp_pre_activation;
        let mlp_activation = args.mlp_activation;
        let mut hidden_nvfp4 = args.hidden_nvfp4;
        let mut tape = args.tape;

        let ln_1 =
            LayerNormWeights::input_from_block(args.layer_norm_module, args.ln_1, args.hidden);
        let hidden =
            forward_layer_norm(&self.ln_1, ln_1, tape.as_mut().map(|tape| &mut tape.ln_1))?;

        let attention_tape = tape.as_mut().map(|tape| AttentionForwardTape {
            qkv_input_nvfp4: tape.qkv_input_nvfp4.reborrow(),
            qkv_f16: &mut *tape.qkv,
            attention_out_f16: &mut *tape.attention_out,
            c_proj_input_nvfp4: tape.c_proj_input_nvfp4.reborrow(),
        });

        let hidden = AttentionWeights::forward(AttentionForwardArgs {
            use_full_attention: args.use_full_attention,
            module: args.attention_module,
            tc_module: args.attention_tc_module,
            quant_module: args.quant_module,
            input_nvfp4: hidden_nvfp4.reborrow(),
            tc_scratch: args.attention_tc_scratch,
            projections: args.projections,
            qkv: &mut *qkv,
            attention_log_sum_exp: &mut *attention_log_sum_exp,
            hidden,
            tape: attention_tape,
        })?;

        if let Some(tape) = tape.as_mut() {
            tape.save_attention_log_sum_exp(hidden.stream, attention_log_sum_exp)?;
        }

        let ln_2 = LayerNormWeights::input_from_block(args.layer_norm_module, args.ln_2, hidden);
        let hidden =
            forward_layer_norm(&self.ln_2, ln_2, tape.as_mut().map(|tape| &mut tape.ln_2))?;

        let mlp_tape = tape.as_mut().map(|tape| crate::types::MlpForwardTape {
            up_input_nvfp4: tape.mlp_up_input_nvfp4.reborrow(),
            down_input_nvfp4: tape.mlp_down_input_nvfp4.reborrow(),
        });

        let hidden = MlpWeights::forward(MlpForwardArgs {
            module: args.mlp_module,
            quant_module: args.quant_module,
            scratch: MlpScratch {
                input_nvfp4: hidden_nvfp4.reborrow(),
                activation_nvfp4: args.mlp_activation_nvfp4,
                pre_activation: &mut *mlp_pre_activation,
                activation: &mut *mlp_activation,
            },
            projections: args.mlp,
            hidden,
            tape: mlp_tape,
        })?;

        save_mlp_tape(
            tape.as_mut(),
            args.attention_tc_module,
            mlp_pre_activation,
            hidden,
        )
    }
}

fn forward_layer_norm<'a>(
    weights: &LayerNormWeights,
    args: LayerNormForwardArgs<'a>,
    tape: Option<&mut LayerNormTape<'_>>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    if let Some(tape) = tape {
        let hidden = weights.forward_save_residual_f16(args, &mut *tape.residual)?;
        tape.save_stats(hidden.stream, hidden.mean, hidden.inv_std)?;
        Ok(hidden)
    } else {
        weights.forward(args)
    }
}

fn save_mlp_tape<'a>(
    tape: Option<&mut crate::types::BlockForwardTape<'_>>,
    f16_module: &F16TcMatmulModule,
    mlp_pre_activation: &DeviceBuffer<f32>,
    hidden: HiddenStateDevice<'a>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    if let Some(tape) = tape {
        tape.save_mlp_up_f16(hidden.stream, f16_module, mlp_pre_activation)?;
    }

    Ok(hidden)
}
