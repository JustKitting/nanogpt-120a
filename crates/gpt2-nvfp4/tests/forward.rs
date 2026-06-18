use std::error::Error;
use std::path::PathBuf;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gpt2_nvfp4::{
    AttentionLogSumExp, GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS, Gpt2, Gpt2BlockWeights,
    Gpt2ForwardArgs, HiddenState, HiddenStateNvfp4, LayerNormTensors, LayerNormWeights, Logits,
    MlpActivation, MlpActivationNvfp4, MlpDownTensors, MlpUpTensors, Nvfp4Shape, Nvfp4Tensor,
    QkvActivation, TokenEmbeddingArgs,
};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::embedding::EmbeddingModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::lm_head::LmHeadModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn gpt2_forward_runs_through_tied_lm_head() -> TestResult {
    run_forward()
}

fn run_forward() -> TestResult {
    let ctx = CudaContext::new(gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module = ctx.load_module_from_file(ptx_path().as_str())?;
    let embedding_module = EmbeddingModule::from_module(module.clone())?;
    let attention_module = AttentionModule::from_module(module.clone())?;
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
        quant_module: &quant_module,
        layer_norm_module: &layer_norm_module,
        mlp_module: &mlp_module,
        lm_head_module: &lm_head_module,
        hidden_nvfp4: HiddenStateNvfp4 {
            bytes: &mut hidden_bytes_dev,
            scales: &mut hidden_scales_dev,
            global_scales: &mut hidden_global_scales_dev,
        },
        mlp_activation_nvfp4: MlpActivationNvfp4 {
            bytes: &mut mlp_activation_bytes_dev,
            scales: &mut mlp_activation_scales_dev,
            global_scales: &mut mlp_activation_global_scales_dev,
        },
        attention_qkv_weights: std::array::from_fn(|i| blocks[i].attn_qkv.weight.mma()),
        attention_qkv_biases: std::array::from_fn(|i| blocks[i].attn_qkv.bias.device()),
        attention_c_proj_weights: std::array::from_fn(|i| blocks[i].attn_c_proj.weight.mma()),
        attention_c_proj_biases: std::array::from_fn(|i| blocks[i].attn_c_proj.bias.device()),
        block_ln_1: std::array::from_fn(|i| blocks[i].ln_1.tensors()),
        block_ln_2: std::array::from_fn(|i| blocks[i].ln_2.tensors()),
        mlp_up: std::array::from_fn(|i| MlpUpTensors {
            weight: blocks[i].mlp_up.weight.mma(),
            bias: blocks[i].mlp_up.bias.device(),
        }),
        mlp_down: std::array::from_fn(|i| MlpDownTensors {
            weight: blocks[i].mlp_down.weight.mma(),
            bias: blocks[i].mlp_down.bias.device(),
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
    assert!(logits.iter().all(|value| value.is_finite()));
    assert!(logits.iter().any(|value| value.abs() > 0.0));
    Ok(())
}

fn token_ids() -> Vec<u32> {
    (0..GPT2_TOKEN_ROWS).map(|i| (i % 127) as u32).collect()
}

struct UploadedNvfp4 {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
}

impl UploadedNvfp4 {
    fn device(&self) -> Nvfp4DeviceTensor<'_> {
        Nvfp4DeviceTensor {
            bytes: &self.bytes,
            scales: &self.scales,
            global_scale: &self.global_scale,
        }
    }

    fn mma(&self) -> Nvfp4FourSixMmaWeightTensor<'_> {
        Nvfp4FourSixMmaWeightTensor {
            bytes: &self.bytes,
            scales: &self.scales,
            global_scale: &self.global_scale,
        }
    }
}

struct UploadedLinear {
    weight: UploadedNvfp4,
    bias: UploadedNvfp4,
}

struct UploadedLayerNorm {
    weight: UploadedNvfp4,
    bias: UploadedNvfp4,
}

impl UploadedLayerNorm {
    fn tensors(&self) -> LayerNormTensors<'_> {
        LayerNormTensors {
            weight: self.weight.device(),
            bias: self.bias.device(),
        }
    }
}

struct UploadedBlock {
    ln_1: UploadedLayerNorm,
    attn_qkv: UploadedLinear,
    attn_c_proj: UploadedLinear,
    ln_2: UploadedLayerNorm,
    mlp_up: UploadedLinear,
    mlp_down: UploadedLinear,
}

fn upload_block(stream: &CudaStream, block: &Gpt2BlockWeights) -> TestResult<UploadedBlock> {
    Ok(UploadedBlock {
        ln_1: upload_layer_norm(stream, &block.ln_1)?,
        attn_qkv: upload_linear(stream, &block.attn.c_attn)?,
        attn_c_proj: upload_linear(stream, &block.attn.c_proj)?,
        ln_2: upload_layer_norm(stream, &block.ln_2)?,
        mlp_up: upload_linear(stream, &block.mlp.c_fc)?,
        mlp_down: upload_linear(stream, &block.mlp.c_proj)?,
    })
}

fn upload_layer_norm(
    stream: &CudaStream,
    layer_norm: &LayerNormWeights,
) -> TestResult<UploadedLayerNorm> {
    Ok(UploadedLayerNorm {
        weight: upload_nvfp4(stream, &layer_norm.weight)?,
        bias: upload_nvfp4(stream, &layer_norm.bias)?,
    })
}

fn upload_linear<W: Nvfp4Shape, B: Nvfp4Shape>(
    stream: &CudaStream,
    linear: &gpt2_nvfp4::LinearWeights<W, B>,
) -> TestResult<UploadedLinear> {
    Ok(UploadedLinear {
        weight: upload_nvfp4(stream, &linear.weight)?,
        bias: upload_nvfp4(stream, &linear.bias)?,
    })
}

fn upload_nvfp4<S: Nvfp4Shape>(
    stream: &CudaStream,
    tensor: &Nvfp4Tensor<S>,
) -> TestResult<UploadedNvfp4> {
    Ok(UploadedNvfp4 {
        bytes: DeviceBuffer::from_host(stream, tensor.bytes.as_ref())?,
        scales: DeviceBuffer::from_host(stream, tensor.scales.as_ref())?,
        global_scale: DeviceBuffer::from_host(stream, &[tensor.global_scale])?,
    })
}

fn gpu_device_index() -> usize {
    std::env::var("CUDA_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn ptx_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../rust_kernels_cuda.ptx")
        .to_string_lossy()
        .into_owned()
}
