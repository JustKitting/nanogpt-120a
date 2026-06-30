use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionLogSumExp, AttentionProjectionTensors, GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS,
    Gpt2, Gpt2ForwardArgs, HiddenState, HiddenStateNvfp4, Logits, MlpActivation,
    MlpActivationNvfp4, MlpDownTensors, MlpProjectionTensors, MlpUpTensors, QkvActivation,
    TokenEmbeddingArgs,
};
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionTcScratch};
use rust_kernels_cuda::embedding::EmbeddingModule;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::lm_head::LmHeadModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

mod common;
#[path = "common/upload.rs"]
mod upload_common;

use common::cuda_test_context;
use upload_common::{TestResult, upload_block, upload_layer_norm, upload_nvfp4};

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
    let mut residual_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let mut normalized_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let mut normalized_amax_dev = DeviceBuffer::<f32>::zeroed(&stream, GPT2_TOKEN_ROWS)?;
    let mut mean_dev = DeviceBuffer::<f32>::zeroed(&stream, GPT2_TOKEN_ROWS)?;
    let mut inv_std_dev = DeviceBuffer::<f32>::zeroed(&stream, GPT2_TOKEN_ROWS)?;
    let mut hidden_bytes_dev = DeviceBuffer::<u8>::zeroed(&stream, HiddenState::LEN / 2)?;
    let mut hidden_scales_dev = DeviceBuffer::<u8>::zeroed(&stream, HiddenState::LEN / 16)?;
    let mut hidden_global_scales_dev = DeviceBuffer::<f32>::zeroed(&stream, GPT2_TOKEN_ROWS)?;
    let mut mlp_pre_activation_dev = DeviceBuffer::<f32>::zeroed(&stream, MlpActivation::LEN)?;
    let mut mlp_activation_dev = DeviceBuffer::<f32>::zeroed(&stream, MlpActivation::LEN)?;
    let mut mlp_activation_bytes_dev = DeviceBuffer::<u8>::zeroed(&stream, MlpActivation::LEN / 2)?;
    let mut mlp_activation_scales_dev =
        DeviceBuffer::<u8>::zeroed(&stream, MlpActivation::LEN / 16)?;
    let mut mlp_activation_global_scales_dev =
        DeviceBuffer::<f32>::zeroed(&stream, GPT2_TOKEN_ROWS)?;
    let mut qkv_dev = DeviceBuffer::<f32>::zeroed(&stream, QkvActivation::LEN)?;
    let mut attention_log_sum_exp_dev =
        DeviceBuffer::<f32>::zeroed(&stream, AttentionLogSumExp::LEN)?;
    let mut tc_q_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let mut tc_k_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let mut tc_v_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let square = GPT2_BATCH_SIZE * gpt2_nvfp4::GPT2_N_HEAD * GPT2_SEQ_LEN * GPT2_SEQ_LEN;
    let mut tc_scores_dev = DeviceBuffer::<f32>::zeroed(&stream, square)?;
    let mut tc_probs_dev = DeviceBuffer::<f32>::zeroed(&stream, square)?;
    let mut tc_out_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let mut tc_chunk_states_dev = DeviceBuffer::<u16>::zeroed(&stream, HiddenState::LEN)?;
    let mut logits_dev = DeviceBuffer::<f32>::zeroed(&stream, Logits::LEN)?;

    model.forward(Gpt2ForwardArgs {
        embeddings: TokenEmbeddingArgs {
            module: &embedding_module,
            stream: &stream,
            tokens: &tokens_dev,
            token_embedding: token_embedding.device(),
            batch_size: GPT2_BATCH_SIZE as u32,
            seq_len: GPT2_SEQ_LEN as u32,
            row_count: GPT2_TOKEN_ROWS as u32,
            residual: &mut residual_dev,
            normalized: &mut normalized_dev,
            normalized_amax: &mut normalized_amax_dev,
            mean: &mut mean_dev,
            inv_std: &mut inv_std_dev,
        },
        attention_module: &attention_module,
        attention_tc_module: &attention_tc_module,
        quant_module: &quant_module,
        layer_norm_module: &layer_norm_module,
        mlp_module: &mlp_module,
        lm_head_module: &lm_head_module,
        hidden_nvfp4: HiddenStateNvfp4 {
            bytes: &mut hidden_bytes_dev,
            scales: &mut hidden_scales_dev,
            global_scales: &mut hidden_global_scales_dev,
        },
        attention_tc_scratch: CausalAttentionTcScratch {
            q: &mut tc_q_dev,
            k: &mut tc_k_dev,
            v: &mut tc_v_dev,
            scores: &mut tc_scores_dev,
            probs: &mut tc_probs_dev,
            compact_out: &mut tc_out_dev,
            chunk_states: &mut tc_chunk_states_dev,
        },
        mlp_activation_nvfp4: MlpActivationNvfp4 {
            bytes: &mut mlp_activation_bytes_dev,
            scales: &mut mlp_activation_scales_dev,
            global_scales: &mut mlp_activation_global_scales_dev,
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
        attention_qkv: &mut qkv_dev,
        attention_log_sum_exp: &mut attention_log_sum_exp_dev,
        mlp_pre_activation: &mut mlp_pre_activation_dev,
        mlp_activation: &mut mlp_activation_dev,
        logits: &mut logits_dev,
        tape: None,
    })?;

    let logits = logits_dev.to_host_vec(&stream)?;
    common::assert_nonzero_finite(&logits);
    Ok(())
}

fn token_ids() -> Vec<u32> {
    (0..GPT2_TOKEN_ROWS).map(|i| (i % 127) as u32).collect()
}
