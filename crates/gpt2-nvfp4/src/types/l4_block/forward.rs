use cuda_core::{DeviceBuffer, DriverError};

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
        let mlp_activation = args.mlp_activation;
        let mut hidden_nvfp4 = args.hidden_nvfp4;
        let mut tape = args.tape;

        if let Some(tape) = tape.as_mut() {
            tape.save_residual_in(args.hidden.stream, args.hidden.residual)?;
        }

        let hidden = self.ln_1.forward(LayerNormWeights::input_from_block(
            args.layer_norm_module,
            args.ln_1,
            args.hidden,
        ))?;

        if let Some(tape) = tape.as_mut() {
            tape.ln_1
                .save(hidden.stream, hidden.residual, hidden.normalized)?;
        }

        let attention_tape = tape.as_mut().map(|tape| AttentionForwardTape {
            qkv_input_nvfp4: tape.qkv_input_nvfp4.reborrow(),
            c_proj_input_nvfp4: tape.c_proj_input_nvfp4.reborrow(),
        });

        let hidden = AttentionWeights::forward(AttentionWeights::input_from_embeddings_with_tape(
            args.attention_module,
            args.quant_module,
            hidden_nvfp4.reborrow(),
            args.projections,
            &mut *qkv,
            hidden,
            attention_tape,
        ))?;

        if let Some(tape) = tape.as_mut() {
            tape.save_qkv(hidden.stream, qkv)?;
            tape.save_attention_out(hidden.stream, hidden.normalized)?;
            tape.save_residual_after_attention(hidden.stream, hidden.residual)?;
        }

        let hidden = self.ln_2.forward(LayerNormWeights::input_from_block(
            args.layer_norm_module,
            args.ln_2,
            hidden,
        ))?;

        if let Some(tape) = tape.as_mut() {
            tape.ln_2
                .save(hidden.stream, hidden.residual, hidden.normalized)?;
        }

        let (pre_activation, mlp_tape) = if let Some(tape) = tape.as_mut() {
            (
                Some(&mut *tape.mlp_up),
                Some(crate::types::MlpForwardTape {
                    up_input_nvfp4: tape.mlp_up_input_nvfp4.reborrow(),
                    down_input_nvfp4: tape.mlp_down_input_nvfp4.reborrow(),
                }),
            )
        } else {
            (None, None)
        };

        let hidden = MlpWeights::forward(MlpWeights::input_from_attention_with_tape(
            args.mlp_module,
            args.quant_module,
            MlpScratch {
                input_nvfp4: hidden_nvfp4.reborrow(),
                activation_nvfp4: args.mlp_activation_nvfp4,
                pre_activation,
                activation: &mut *mlp_activation,
            },
            MlpProjectionTensors {
                up: args.mlp_up,
                down: args.mlp_down,
            },
            hidden,
            mlp_tape,
        ))?;

        save_mlp_tape(tape.as_mut(), mlp_activation, hidden)
    }
}

fn save_mlp_tape<'a>(
    tape: Option<&mut crate::types::BlockForwardTape<'_>>,
    mlp_activation: &DeviceBuffer<f32>,
    hidden: HiddenStateDevice<'a>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    if let Some(tape) = tape {
        tape.save_mlp_relu2(hidden.stream, mlp_activation)?;
        tape.save_residual_out(hidden.stream, hidden.residual)?;
    }

    Ok(hidden)
}
