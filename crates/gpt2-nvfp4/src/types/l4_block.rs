use crate::random::InitRng;
use cuda_core::{DeviceBuffer, DriverError};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use super::{
    AttentionProjectionTensors, AttentionWeights, BlockForwardTape, HiddenStateDevice,
    HiddenStateNvfp4, LayerNormTensors, LayerNormWeights, MlpActivationNvfp4, MlpDownTensors,
    MlpProjectionTensors, MlpScratch, MlpUpTensors, MlpWeights,
};

pub struct BlockForwardArgs<'a, 'scratch> {
    pub attention_module: &'a AttentionModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub layer_norm_module: &'a LayerNormModule,
    pub mlp_module: &'a MlpModule,
    pub hidden_nvfp4: HiddenStateNvfp4<'scratch>,
    pub mlp_activation_nvfp4: MlpActivationNvfp4<'scratch>,
    pub projections: AttentionProjectionTensors<'a>,
    pub ln_1: LayerNormTensors<'a>,
    pub ln_2: LayerNormTensors<'a>,
    pub mlp_up: MlpUpTensors<'a>,
    pub mlp_down: MlpDownTensors<'a>,
    pub qkv: &'scratch mut DeviceBuffer<f32>,
    pub mlp_activation: &'scratch mut DeviceBuffer<f32>,
    pub hidden: HiddenStateDevice<'a>,
    pub tape: Option<BlockForwardTape<'scratch>>,
}

#[derive(Clone, Debug)]
pub struct Gpt2BlockWeights {
    pub ln_1: LayerNormWeights,
    pub attn: AttentionWeights,
    pub ln_2: LayerNormWeights,
    pub mlp: MlpWeights,
}

impl Gpt2BlockWeights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        Self {
            ln_1: LayerNormWeights::init(),
            attn: AttentionWeights::init(rng),
            ln_2: LayerNormWeights::init(),
            mlp: MlpWeights::init(rng),
        }
    }

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
            tape.ln_1.save(
                hidden.stream,
                hidden.residual,
                hidden.normalized,
                hidden.normalized_amax,
            )?;
        }

        let hidden = AttentionWeights::forward(AttentionWeights::input_from_embeddings(
            args.attention_module,
            args.quant_module,
            hidden_nvfp4.reborrow(),
            args.projections,
            &mut *qkv,
            hidden,
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
            tape.ln_2.save(
                hidden.stream,
                hidden.residual,
                hidden.normalized,
                hidden.normalized_amax,
            )?;
        }

        MlpWeights::forward(MlpWeights::input_from_attention(
            args.mlp_module,
            args.quant_module,
            MlpScratch {
                input_nvfp4: hidden_nvfp4.reborrow(),
                activation_nvfp4: args.mlp_activation_nvfp4,
                activation: &mut *mlp_activation,
            },
            MlpProjectionTensors {
                up: args.mlp_up,
                down: args.mlp_down,
            },
            hidden,
        ))
        .and_then(|hidden| {
            if let Some(tape) = tape.as_mut() {
                tape.save_mlp_activation(hidden.stream, mlp_activation)?;
                tape.save_residual_out(hidden.stream, hidden.residual)?;
            }

            Ok(hidden)
        })
    }
}
