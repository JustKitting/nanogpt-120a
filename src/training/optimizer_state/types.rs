use cuda_core::DeviceBuffer;
use gpt2_nvfp4::GPT2_N_LAYER;

use super::UpdateSkipState;

pub struct OptimizerStateBuffers {
    pub(in crate::training) step: u32,
    pub(in crate::training) schedule_free_weight_sum: f32,
    pub(in crate::training) update_skip: UpdateSkipState,
    pub(in crate::training) token_embedding: AdamState,
    pub(in crate::training) ln_f: LayerNormState,
    pub(in crate::training) next_latent: NextLatState,
    pub(in crate::training) blocks: [BlockState; GPT2_N_LAYER],
}

pub(in crate::training) struct BlockState {
    pub(in crate::training) ln_1: LayerNormState,
    pub(in crate::training) attn_qkv: LinearState,
    pub(in crate::training) attn_c_proj: LinearState,
    pub(in crate::training) ln_2: LayerNormState,
    pub(in crate::training) mlp_up: LinearState,
    pub(in crate::training) mlp_down: LinearState,
}

pub(in crate::training) struct NextLatState {
    pub(in crate::training) norm: LayerNormState,
    pub(in crate::training) input_projection: LinearState,
    pub(in crate::training) transition: LinearState,
    pub(in crate::training) output_projection: LinearState,
}

pub(in crate::training) struct LayerNormState {
    pub(in crate::training) weight: AdamState,
    pub(in crate::training) bias: AdamState,
}

pub(in crate::training) struct LinearState {
    pub(in crate::training) weight_aurora: AuroraState,
    pub(in crate::training) bias: AdamState,
}

pub(in crate::training) struct AdamState {
    pub(in crate::training) z_master: DeviceBuffer<f32>,
    pub(in crate::training) x_master: DeviceBuffer<f32>,
    pub(in crate::training) first: DeviceBuffer<f32>,
    pub(in crate::training) second: DeviceBuffer<f32>,
}

pub(in crate::training) struct AuroraState {
    pub(in crate::training) z_master: DeviceBuffer<f32>,
    pub(in crate::training) x_master: DeviceBuffer<f32>,
    pub(in crate::training) momentum: DeviceBuffer<f32>,
}
