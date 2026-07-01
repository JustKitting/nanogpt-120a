use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;

use super::args::BlockForwardArgs;
use super::weights::Gpt2BlockWeights;
use crate::types::{
    AttentionForwardArgs, AttentionWeights, HiddenStateDevice, LayerNormWeights, MlpForwardArgs,
    MlpScratch, MlpWeights,
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
        let hidden = self
            .ln_1
            .forward_with_tape(ln_1, tape.as_mut().map(|tape| &mut tape.ln_1))?;

        let attention_tape = tape.as_mut().map(|tape| tape.attention_forward());

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
        let hidden = self
            .ln_2
            .forward_with_tape(ln_2, tape.as_mut().map(|tape| &mut tape.ln_2))?;

        let mlp_tape = tape.as_mut().map(|tape| tape.mlp_forward());

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
