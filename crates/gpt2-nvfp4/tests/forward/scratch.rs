use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionLogSumExp, HiddenState, Logits, MlpActivation, QkvActivation, GPT2_BATCH_SIZE,
    GPT2_N_HEAD, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS,
};

const ATTENTION_SQUARE: usize = GPT2_BATCH_SIZE * GPT2_N_HEAD * GPT2_SEQ_LEN * GPT2_SEQ_LEN;

macro_rules! forward_scratch {
    ($($name:ident: $ty:ty = $len:expr),+ $(,)?) => {
        pub struct ForwardScratch {
            $(pub $name: DeviceBuffer<$ty>,)+
        }

        impl ForwardScratch {
            pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
                Ok(Self {
                    $($name: DeviceBuffer::<$ty>::zeroed(stream, $len)?,)+
                })
            }
        }
    };
}

forward_scratch! {
    residual: f32 = HiddenState::LEN,
    normalized: f32 = HiddenState::LEN,
    normalized_amax: f32 = GPT2_TOKEN_ROWS,
    mean: f32 = GPT2_TOKEN_ROWS,
    inv_std: f32 = GPT2_TOKEN_ROWS,
    hidden_bytes: u8 = HiddenState::LEN / 2,
    hidden_scales: u8 = HiddenState::LEN / 16,
    hidden_global_scales: f32 = GPT2_TOKEN_ROWS,
    mlp_pre_activation: f32 = MlpActivation::LEN,
    mlp_activation: f32 = MlpActivation::LEN,
    mlp_activation_bytes: u8 = MlpActivation::LEN / 2,
    mlp_activation_scales: u8 = MlpActivation::LEN / 16,
    mlp_activation_global_scales: f32 = GPT2_TOKEN_ROWS,
    qkv: f32 = QkvActivation::LEN,
    attention_log_sum_exp: f32 = AttentionLogSumExp::LEN,
    tc_q: f32 = HiddenState::LEN,
    tc_k: f32 = HiddenState::LEN,
    tc_v: f32 = HiddenState::LEN,
    tc_scores: f32 = ATTENTION_SQUARE,
    tc_probs: f32 = ATTENTION_SQUARE,
    tc_out: f32 = HiddenState::LEN,
    tc_chunk_states: u16 = HiddenState::LEN,
    logits: f32 = Logits::LEN,
}
