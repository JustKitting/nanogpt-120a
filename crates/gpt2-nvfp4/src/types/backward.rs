use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use crate::GPT2_N_LAYER;

pub struct Gpt2BackwardContext<'a> {
    pub saved: Gpt2ForwardSaved<'a>,
    pub grads: Gpt2BackwardGrads<'a>,
}

#[derive(Clone, Copy)]
pub struct Gpt2ForwardSaved<'a> {
    pub tokens: &'a DeviceBuffer<u32>,
    pub batch_size: u32,
    pub seq_len: u32,
    pub row_count: u32,
    pub blocks: [BlockForwardSaved<'a>; GPT2_N_LAYER],
    pub final_norm: LayerNormSaved<'a>,
    pub lm_head_input_nvfp4: Nvfp4RowwiseDeviceTensor<'a>,
    pub logits: &'a DeviceBuffer<f32>,
}

#[derive(Clone, Copy)]
pub struct BlockForwardSaved<'a> {
    pub batch_size: u32,
    pub seq_len: u32,
    pub row_count: u32,
    pub ln_1: LayerNormSaved<'a>,
    pub qkv_input_nvfp4: Nvfp4RowwiseDeviceTensor<'a>,
    pub qkv: &'a DeviceBuffer<u16>,
    pub attention_out: &'a DeviceBuffer<u16>,
    pub attention_log_sum_exp: &'a DeviceBuffer<f32>,
    pub c_proj_input_nvfp4: Nvfp4RowwiseDeviceTensor<'a>,
    pub ln_2: LayerNormSaved<'a>,
    pub mlp_up_input_nvfp4: Nvfp4RowwiseDeviceTensor<'a>,
    pub mlp_up: &'a DeviceBuffer<u16>,
    pub mlp_down_input_nvfp4: Nvfp4RowwiseDeviceTensor<'a>,
}

#[derive(Clone, Copy)]
pub struct LayerNormSaved<'a> {
    pub row_count: u32,
    pub residual: &'a DeviceBuffer<u16>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
}

pub struct Gpt2BackwardGrads<'a> {
    pub dlogits: &'a mut DeviceBuffer<f32>,
    pub d_embedding_residual: &'a mut DeviceBuffer<f32>,
    pub blocks: [BlockBackwardGrads<'a>; GPT2_N_LAYER],
    pub final_norm: LayerNormGrads<'a>,
}

pub struct BlockBackwardGrads<'a> {
    pub d_residual_in: &'a mut DeviceBuffer<f32>,
    pub ln_1: LayerNormGrads<'a>,
    pub d_qkv: &'a mut DeviceBuffer<f32>,
    pub d_attention_out: &'a mut DeviceBuffer<f32>,
    pub d_residual_after_attention: &'a mut DeviceBuffer<f32>,
    pub ln_2: LayerNormGrads<'a>,
    pub d_mlp_up: &'a mut DeviceBuffer<f32>,
    pub d_mlp_relu2: &'a mut DeviceBuffer<f32>,
    pub d_attn_qkv_weight: &'a mut DeviceBuffer<f32>,
    pub d_attn_qkv_bias: &'a mut DeviceBuffer<f32>,
    pub d_attn_c_proj_weight: &'a mut DeviceBuffer<f32>,
    pub d_attn_c_proj_bias: &'a mut DeviceBuffer<f32>,
    pub d_mlp_c_fc_weight: &'a mut DeviceBuffer<f32>,
    pub d_mlp_c_fc_bias: &'a mut DeviceBuffer<f32>,
    pub d_mlp_c_proj_weight: &'a mut DeviceBuffer<f32>,
    pub d_mlp_c_proj_bias: &'a mut DeviceBuffer<f32>,
    pub d_residual_out: &'a mut DeviceBuffer<f32>,
}

pub struct LayerNormGrads<'a> {
    pub d_residual: &'a mut DeviceBuffer<f32>,
    pub d_normalized: &'a mut DeviceBuffer<f32>,
    pub d_weight: &'a mut DeviceBuffer<f32>,
    pub d_bias: &'a mut DeviceBuffer<f32>,
}

impl<'a> BlockBackwardGrads<'a> {
    pub fn reborrow(&mut self) -> BlockBackwardGrads<'_> {
        BlockBackwardGrads {
            d_residual_in: &mut *self.d_residual_in,
            ln_1: self.ln_1.reborrow(),
            d_qkv: &mut *self.d_qkv,
            d_attention_out: &mut *self.d_attention_out,
            d_residual_after_attention: &mut *self.d_residual_after_attention,
            ln_2: self.ln_2.reborrow(),
            d_mlp_up: &mut *self.d_mlp_up,
            d_mlp_relu2: &mut *self.d_mlp_relu2,
            d_attn_qkv_weight: &mut *self.d_attn_qkv_weight,
            d_attn_qkv_bias: &mut *self.d_attn_qkv_bias,
            d_attn_c_proj_weight: &mut *self.d_attn_c_proj_weight,
            d_attn_c_proj_bias: &mut *self.d_attn_c_proj_bias,
            d_mlp_c_fc_weight: &mut *self.d_mlp_c_fc_weight,
            d_mlp_c_fc_bias: &mut *self.d_mlp_c_fc_bias,
            d_mlp_c_proj_weight: &mut *self.d_mlp_c_proj_weight,
            d_mlp_c_proj_bias: &mut *self.d_mlp_c_proj_bias,
            d_residual_out: &mut *self.d_residual_out,
        }
    }

    pub fn reborrow_with_residual_in<'b>(
        &'b mut self,
        d_residual_in: &'b mut DeviceBuffer<f32>,
    ) -> BlockBackwardGrads<'b> {
        BlockBackwardGrads {
            d_residual_in,
            ln_1: self.ln_1.reborrow(),
            d_qkv: &mut *self.d_qkv,
            d_attention_out: &mut *self.d_attention_out,
            d_residual_after_attention: &mut *self.d_residual_after_attention,
            ln_2: self.ln_2.reborrow(),
            d_mlp_up: &mut *self.d_mlp_up,
            d_mlp_relu2: &mut *self.d_mlp_relu2,
            d_attn_qkv_weight: &mut *self.d_attn_qkv_weight,
            d_attn_qkv_bias: &mut *self.d_attn_qkv_bias,
            d_attn_c_proj_weight: &mut *self.d_attn_c_proj_weight,
            d_attn_c_proj_bias: &mut *self.d_attn_c_proj_bias,
            d_mlp_c_fc_weight: &mut *self.d_mlp_c_fc_weight,
            d_mlp_c_fc_bias: &mut *self.d_mlp_c_fc_bias,
            d_mlp_c_proj_weight: &mut *self.d_mlp_c_proj_weight,
            d_mlp_c_proj_bias: &mut *self.d_mlp_c_proj_bias,
            d_residual_out: &mut *self.d_residual_out,
        }
    }
}

impl<'a> LayerNormGrads<'a> {
    pub fn reborrow(&mut self) -> LayerNormGrads<'_> {
        LayerNormGrads {
            d_residual: &mut *self.d_residual,
            d_normalized: &mut *self.d_normalized,
            d_weight: &mut *self.d_weight,
            d_bias: &mut *self.d_bias,
        }
    }
}
