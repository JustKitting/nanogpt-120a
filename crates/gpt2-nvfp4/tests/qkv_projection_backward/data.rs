use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionBackwardSeeds, BlockForwardSaved, GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS,
    Gpt2Rng, HiddenState, LayerNormSaved, Nvfp4Shape, QkvActivation, QkvWeightShape,
};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

const E2M1_MIN_PAIR: u8 = 0x11;
const E4M3_ONE: u8 = 0x38;

pub fn saved_block<'a>(
    qkv_input_bytes: &'a DeviceBuffer<u8>,
    qkv_input_scales: &'a DeviceBuffer<u8>,
    qkv_input_globals: &'a DeviceBuffer<f32>,
    dummy: &'a DeviceBuffer<f32>,
    dummy_u16: &'a DeviceBuffer<u16>,
) -> BlockForwardSaved<'a> {
    let rowwise =
        Nvfp4RowwiseDeviceTensor::new(qkv_input_bytes, qkv_input_scales, qkv_input_globals);
    let layer_norm = LayerNormSaved {
        row_count: GPT2_TOKEN_ROWS as u32,
        residual: dummy_u16,
        mean: dummy,
        inv_std: dummy,
    };

    BlockForwardSaved {
        batch_size: GPT2_BATCH_SIZE as u32,
        seq_len: GPT2_SEQ_LEN as u32,
        row_count: GPT2_TOKEN_ROWS as u32,
        ln_1: layer_norm,
        qkv_input_nvfp4: rowwise,
        qkv: dummy_u16,
        attention_out: dummy_u16,
        attention_log_sum_exp: dummy,
        c_proj_input_nvfp4: rowwise,
        ln_2: layer_norm,
        mlp_up_input_nvfp4: rowwise,
        mlp_up: dummy_u16,
        mlp_down_input_nvfp4: rowwise,
    }
}

pub fn qkv_input_bytes() -> Vec<u8> {
    vec![E2M1_MIN_PAIR; HiddenState::LEN / 2]
}

pub fn hidden_scales() -> Vec<u8> {
    vec![E4M3_ONE; HiddenState::LEN / 16]
}

pub fn row_global_scales() -> Vec<f32> {
    vec![1.0; GPT2_TOKEN_ROWS]
}

pub fn qkv_weight_bytes() -> Vec<u8> {
    vec![E2M1_MIN_PAIR; QkvWeightShape::BYTE_LEN]
}

pub fn qkv_weight_scales() -> Vec<u8> {
    vec![E4M3_ONE; QkvWeightShape::SCALE_LEN]
}

pub fn zero_bytes() -> Vec<u8> {
    vec![0; QkvWeightShape::BYTE_LEN]
}

pub fn one_scales() -> Vec<u8> {
    vec![E4M3_ONE; QkvWeightShape::SCALE_LEN]
}

pub fn d_qkv_values() -> Vec<f32> {
    (0..QkvActivation::LEN)
        .map(|index| 0.000_244_140_63 * ((index % 11) as f32 + 1.0))
        .collect()
}

pub fn seeds() -> AttentionBackwardSeeds {
    let mut rng = Gpt2Rng::new(0x5156_4b56);
    AttentionBackwardSeeds::from_rng(&mut rng)
}
