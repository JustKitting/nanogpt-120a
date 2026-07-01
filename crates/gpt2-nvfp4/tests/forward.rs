use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionProjectionTensors, Gpt2, Gpt2ForwardArgs, HiddenStateNvfp4, MlpActivationNvfp4,
    MlpDownTensors, MlpProjectionTensors, MlpUpTensors, TokenEmbeddingArgs, GPT2_BATCH_SIZE,
    GPT2_SEQ_LEN, GPT2_TOKEN_ROWS,
};
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionTcScratch};
use rust_kernels_cuda::embedding::EmbeddingModule;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::lm_head::LmHeadModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

mod common;
#[path = "forward/scratch.rs"]
mod scratch;
#[path = "common/upload.rs"]
mod upload_common;

use common::cuda_test_context;
use scratch::ForwardScratch;
use upload_common::{upload_block, upload_layer_norm, upload_nvfp4, TestResult};

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn gpt2_forward_runs_through_tied_lm_head() -> TestResult {
    let (_, stream, module) = cuda_test_context()?;
    let embedding_module = EmbeddingModule::from_module(module.clone())?;
    let attention_module = AttentionModule::from_module(module.clone())?;
    let attention_tc_module = F16TcMatmulModule::from_module(module.clone())?;
    let quant_module = Nvfp4QuantModule::from_module(module.clone())?;
    let layer_norm_module = LayerNormModule::from_module(module.clone())?;
    let mlp_module = MlpModule::from_module(module.clone())?;
    let lm_head_module = LmHeadModule::from_module(module)?;

    let mut model = Gpt2::new();
    model.init(0x4750_5432);
    let weights = model
        .weights()
        .expect("Gpt2::init must create model weights");

    let token_embedding = upload_nvfp4(&stream, &weights.embeddings.wte)?;
    let blocks = weights
        .h
        .iter()
        .map(|block| upload_block(&stream, block))
        .collect::<TestResult<Vec<_>>>()?;
    let ln_f = upload_layer_norm(&stream, &weights.ln_f)?;

    let tokens = token_ids();
    let tokens_dev = DeviceBuffer::from_host(&stream, &tokens)?;
    let mut scratch = ForwardScratch::new(&stream)?;

    model.forward(Gpt2ForwardArgs {
        embeddings: TokenEmbeddingArgs {
            module: &embedding_module,
            stream: &stream,
            tokens: &tokens_dev,
            token_embedding: token_embedding.device(),
            batch_size: GPT2_BATCH_SIZE as u32,
            seq_len: GPT2_SEQ_LEN as u32,
            row_count: GPT2_TOKEN_ROWS as u32,
            residual: &mut scratch.residual,
            normalized: &mut scratch.normalized,
            normalized_amax: &mut scratch.normalized_amax,
            mean: &mut scratch.mean,
            inv_std: &mut scratch.inv_std,
        },
        attention_module: &attention_module,
        attention_tc_module: &attention_tc_module,
        quant_module: &quant_module,
        layer_norm_module: &layer_norm_module,
        mlp_module: &mlp_module,
        lm_head_module: &lm_head_module,
        hidden_nvfp4: HiddenStateNvfp4 {
            bytes: &mut scratch.hidden_bytes,
            scales: &mut scratch.hidden_scales,
            global_scales: &mut scratch.hidden_global_scales,
        },
        attention_tc_scratch: CausalAttentionTcScratch {
            q: &mut scratch.tc_q,
            k: &mut scratch.tc_k,
            v: &mut scratch.tc_v,
            scores: &mut scratch.tc_scores,
            probs: &mut scratch.tc_probs,
            compact_out: &mut scratch.tc_out,
            chunk_states: &mut scratch.tc_chunk_states,
        },
        mlp_activation_nvfp4: MlpActivationNvfp4 {
            bytes: &mut scratch.mlp_activation_bytes,
            scales: &mut scratch.mlp_activation_scales,
            global_scales: &mut scratch.mlp_activation_global_scales,
        },
        attention: std::array::from_fn(|i| AttentionProjectionTensors {
            qkv_weight: blocks[i].attn_qkv.weight.mma(),
            qkv_bias: blocks[i].attn_qkv.bias.device(),
            c_proj_weight: blocks[i].attn_c_proj.weight.mma(),
            c_proj_bias: blocks[i].attn_c_proj.bias.device(),
        }),
        block_ln_1: std::array::from_fn(|i| blocks[i].ln_1.tensors()),
        block_ln_2: std::array::from_fn(|i| blocks[i].ln_2.tensors()),
        mlp: std::array::from_fn(|i| MlpProjectionTensors {
            up: MlpUpTensors {
                weight: blocks[i].mlp_up.weight.mma(),
                bias: blocks[i].mlp_up.bias.device(),
            },
            down: MlpDownTensors {
                weight: blocks[i].mlp_down.weight.mma(),
                bias: blocks[i].mlp_down.bias.device(),
            },
        }),
        ln_f: ln_f.tensors(),
        attention_qkv: &mut scratch.qkv,
        attention_log_sum_exp: &mut scratch.attention_log_sum_exp,
        mlp_pre_activation: &mut scratch.mlp_pre_activation,
        mlp_activation: &mut scratch.mlp_activation,
        logits: &mut scratch.logits,
        tape: None,
    })?;

    let logits = scratch.logits.to_host_vec(&stream)?;
    common::assert_nonzero_finite(&logits);
    Ok(())
}

fn token_ids() -> Vec<u32> {
    (0..GPT2_TOKEN_ROWS).map(|i| (i % 127) as u32).collect()
}
