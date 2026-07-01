use cuda_core::DeviceBuffer;

use super::LayerNormGrads;

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

macro_rules! reborrow_block_grads {
    ($self:expr, $d_residual_in:expr) => {
        BlockBackwardGrads {
            d_residual_in: $d_residual_in,
            ln_1: $self.ln_1.reborrow(),
            d_qkv: &mut *$self.d_qkv,
            d_attention_out: &mut *$self.d_attention_out,
            d_residual_after_attention: &mut *$self.d_residual_after_attention,
            ln_2: $self.ln_2.reborrow(),
            d_mlp_up: &mut *$self.d_mlp_up,
            d_mlp_relu2: &mut *$self.d_mlp_relu2,
            d_attn_qkv_weight: &mut *$self.d_attn_qkv_weight,
            d_attn_qkv_bias: &mut *$self.d_attn_qkv_bias,
            d_attn_c_proj_weight: &mut *$self.d_attn_c_proj_weight,
            d_attn_c_proj_bias: &mut *$self.d_attn_c_proj_bias,
            d_mlp_c_fc_weight: &mut *$self.d_mlp_c_fc_weight,
            d_mlp_c_fc_bias: &mut *$self.d_mlp_c_fc_bias,
            d_mlp_c_proj_weight: &mut *$self.d_mlp_c_proj_weight,
            d_mlp_c_proj_bias: &mut *$self.d_mlp_c_proj_bias,
            d_residual_out: &mut *$self.d_residual_out,
        }
    };
}

impl<'a> BlockBackwardGrads<'a> {
    pub fn reborrow(&mut self) -> BlockBackwardGrads<'_> {
        reborrow_block_grads!(self, &mut *self.d_residual_in)
    }

    pub fn reborrow_with_residual_in<'b>(
        &'b mut self,
        d_residual_in: &'b mut DeviceBuffer<f32>,
    ) -> BlockBackwardGrads<'b> {
        reborrow_block_grads!(self, d_residual_in)
    }
}
