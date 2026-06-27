use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;

use super::args::BlockForwardArgs;
use super::weights::Gpt2BlockWeights;
use crate::types::{
    AttentionForwardTape, AttentionWeights, HiddenStateDevice, LayerNormWeights,
    MlpProjectionTensors, MlpScratch, MlpWeights,
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
        let hidden = if let Some(tape) = tape.as_mut() {
            let hidden = self
                .ln_1
                .forward_save_residual_f16(ln_1, &mut *tape.ln_1.residual)?;
            tape.ln_1
                .save_stats(hidden.stream, hidden.mean, hidden.inv_std)?;
            hidden
        } else {
            self.ln_1.forward(ln_1)?
        };

        let attention_tape = tape.as_mut().map(|tape| AttentionForwardTape {
            qkv_input_nvfp4: tape.qkv_input_nvfp4.reborrow(),
            qkv_f16: &mut *tape.qkv,
            attention_out_f16: &mut *tape.attention_out,
            c_proj_input_nvfp4: tape.c_proj_input_nvfp4.reborrow(),
        });

        let hidden = AttentionWeights::forward(AttentionWeights::input_from_embeddings_with_tape(
            args.use_full_attention,
            args.attention_module,
            args.attention_tc_module,
            args.quant_module,
            hidden_nvfp4.reborrow(),
            args.attention_tc_scratch,
            args.projections,
            &mut *qkv,
            &mut *attention_log_sum_exp,
            hidden,
            attention_tape,
        ))?;

        if let Some(tape) = tape.as_mut() {
            tape.save_attention_log_sum_exp(hidden.stream, attention_log_sum_exp)?;
        }

        let ln_2 = LayerNormWeights::input_from_block(args.layer_norm_module, args.ln_2, hidden);
        let hidden = if let Some(tape) = tape.as_mut() {
            let hidden = self
                .ln_2
                .forward_save_residual_f16(ln_2, &mut *tape.ln_2.residual)?;
            tape.ln_2
                .save_stats(hidden.stream, hidden.mean, hidden.inv_std)?;
            hidden
        } else {
            self.ln_2.forward(ln_2)?
        };

        let mlp_tape = tape.as_mut().map(|tape| crate::types::MlpForwardTape {
            up_input_nvfp4: tape.mlp_up_input_nvfp4.reborrow(),
            down_input_nvfp4: tape.mlp_down_input_nvfp4.reborrow(),
        });

        let hidden = MlpWeights::forward(MlpWeights::input_from_attention_with_tape(
            args.mlp_module,
            args.quant_module,
            MlpScratch {
                input_nvfp4: hidden_nvfp4.reborrow(),
                activation_nvfp4: args.mlp_activation_nvfp4,
                pre_activation: &mut *mlp_pre_activation,
                activation: &mut *mlp_activation,
            },
            MlpProjectionTensors {
                up: args.mlp_up,
                down: args.mlp_down,
            },
            hidden,
            mlp_tape,
        ))?;

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
