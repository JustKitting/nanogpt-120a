use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionLogSumExp, BlockForwardSaved, GPT2_BATCH_SIZE, GPT2_N_HEAD, GPT2_SEQ_LEN,
    GPT2_TOKEN_ROWS, HiddenState, LayerNormSaved,
};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

pub fn saved_block<'a>(
    qkv: &'a DeviceBuffer<f32>,
    attention_out: &'a DeviceBuffer<f32>,
    attention_log_sum_exp: &'a DeviceBuffer<f32>,
    dummy: &'a DeviceBuffer<f32>,
    dummy_bytes: &'a DeviceBuffer<u8>,
    dummy_scales: &'a DeviceBuffer<u8>,
    dummy_global_scales: &'a DeviceBuffer<f32>,
) -> BlockForwardSaved<'a> {
    let rowwise = Nvfp4RowwiseDeviceTensor {
        bytes: dummy_bytes,
        scales: dummy_scales,
        global_scales: dummy_global_scales,
    };
    let layer_norm = LayerNormSaved {
        row_count: GPT2_TOKEN_ROWS as u32,
        residual: dummy,
        normalized: dummy,
        mean: dummy,
        inv_std: dummy,
    };

    BlockForwardSaved {
        batch_size: GPT2_BATCH_SIZE as u32,
        seq_len: GPT2_SEQ_LEN as u32,
        row_count: GPT2_TOKEN_ROWS as u32,
        residual_in: dummy,
        ln_1: layer_norm,
        qkv_input_nvfp4: rowwise,
        qkv,
        attention_out,
        attention_log_sum_exp,
        c_proj_input_nvfp4: rowwise,
        residual_after_attention: dummy,
        ln_2: layer_norm,
        mlp_up_input_nvfp4: rowwise,
        mlp_up: dummy,
        mlp_relu2: dummy,
        mlp_down_input_nvfp4: rowwise,
        residual_out: dummy,
    }
}

pub fn d_out_values() -> Vec<f32> {
    (0..HiddenState::LEN)
        .map(|index| (index % 17) as f32 * 0.000_244_140_63)
        .collect()
}

pub fn log_sum_exp_values() -> Vec<f32> {
    let mut log_sum_exp = vec![0.0_f32; AttentionLogSumExp::LEN];
    for batch in 0..GPT2_BATCH_SIZE {
        for head in 0..GPT2_N_HEAD {
            for token in 0..GPT2_SEQ_LEN {
                let index = (batch * GPT2_N_HEAD + head) * GPT2_SEQ_LEN + token;
                log_sum_exp[index] = ((token + 1) as f32).ln();
            }
        }
    }
    log_sum_exp
}
