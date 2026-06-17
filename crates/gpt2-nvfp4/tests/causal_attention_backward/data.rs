use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionLse, BlockForwardSaved, GPT2_CONTEXT_LEN, GPT2_N_HEAD, HiddenState, LayerNormSaved,
};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

pub fn saved_block<'a>(
    qkv: &'a DeviceBuffer<f32>,
    attention_out: &'a DeviceBuffer<f32>,
    attention_lse: &'a DeviceBuffer<f32>,
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
        residual: dummy,
        normalized: dummy,
        mean: dummy,
        inv_std: dummy,
    };

    BlockForwardSaved {
        residual_in: dummy,
        ln_1: layer_norm,
        qkv_input_nvfp4: rowwise,
        qkv,
        attention_out,
        attention_lse,
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

pub fn lse_values() -> Vec<f32> {
    let mut lse = vec![0.0_f32; AttentionLse::LEN];
    for head in 0..GPT2_N_HEAD {
        for token in 0..GPT2_CONTEXT_LEN {
            lse[head * GPT2_CONTEXT_LEN + token] = ((token + 1) as f32).ln();
        }
    }
    lse
}
