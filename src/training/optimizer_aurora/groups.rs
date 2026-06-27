use super::super::grads::BackwardBuffers;
use super::super::next_latent::NextLatGradBuffers;
use super::super::optimizer_state::OptimizerStateBuffers;
use crate::upload::UploadedModel;
use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    GPT2_FULL_ATTENTION_QKV, GPT2_MLP, GPT2_N_EMBD, GPT2_N_LAYER, GPT2_QKV, NEXTLAT_HIDDEN,
    NEXTLAT_INPUT, uses_full_attention,
};
use rust_kernels_cuda::optimizer::{AURORA_MATRIX_PHASES, AuroraSlotDescriptor};

mod padding;
mod ptrs;
mod table;
use padding::AuroraPaddingBuffers;
use table::upload_table;

pub(in crate::training) struct AuroraPointerTables {
    pub(in crate::training) all: AuroraGroupTable,
    pub(in crate::training) slot_count: usize,
    _padding: AuroraPaddingBuffers,
}

pub(in crate::training) struct AuroraGroupTable {
    pub(super) slots: DeviceBuffer<AuroraSlotDescriptor>,
}

#[derive(Clone, Copy)]
pub(super) struct HostPtrs {
    grad: u64,
    momentum: u64,
    z_master: u64,
    x_master: u64,
    bytes: u64,
    scales: u64,
    global_scale: u64,
    rows: u32,
    cols: u32,
    learning_rate_multiplier: f32,
}

impl AuroraPointerTables {
    pub(in crate::training) fn new(
        stream: &CudaStream,
        uploaded: &UploadedModel,
        grads: &BackwardBuffers,
        next_latent_grads: &NextLatGradBuffers,
        state: &OptimizerStateBuffers,
    ) -> Result<Self, DriverError> {
        let padding = AuroraPaddingBuffers::new(stream)?;
        let mut rows = all_slots(uploaded, grads, next_latent_grads, state);
        schedule_slots(&mut rows);
        pad_slots(&mut rows, &padding);
        Ok(Self {
            all: upload_table(stream, &rows)?,
            slot_count: rows.len(),
            _padding: padding,
        })
    }
}

fn all_slots(
    uploaded: &UploadedModel,
    grads: &BackwardBuffers,
    next_latent_grads: &NextLatGradBuffers,
    state: &OptimizerStateBuffers,
) -> Vec<HostPtrs> {
    let mut rows = Vec::with_capacity(GPT2_N_LAYER * 4 + 3);
    for i in 0..GPT2_N_LAYER {
        let qkv_dim = if uses_full_attention(i) {
            GPT2_FULL_ATTENTION_QKV
        } else {
            GPT2_QKV
        };
        rows.push(ptrs::qkv(uploaded, grads, state, i).shape(GPT2_N_EMBD, qkv_dim));
    }
    append(&mut rows, GPT2_N_EMBD, GPT2_N_EMBD, |i| {
        ptrs::c_proj(uploaded, grads, state, i)
    });
    append(&mut rows, GPT2_N_EMBD, GPT2_MLP, |i| {
        ptrs::mlp_up(uploaded, grads, state, i)
    });
    append(&mut rows, GPT2_MLP, GPT2_N_EMBD, |i| {
        ptrs::mlp_down(uploaded, grads, state, i)
    });
    rows.push(
        ptrs::next_latent_input_projection(uploaded, next_latent_grads, state)
            .learning_rate_multiplier(super::super::learning_rate::next_latent_scale())
            .shape(NEXTLAT_INPUT, NEXTLAT_HIDDEN),
    );
    rows.push(
        ptrs::next_latent_transition(uploaded, next_latent_grads, state)
            .learning_rate_multiplier(super::super::learning_rate::next_latent_scale())
            .shape(NEXTLAT_HIDDEN, NEXTLAT_HIDDEN),
    );
    rows.push(
        ptrs::next_latent_output_projection(uploaded, next_latent_grads, state)
            .learning_rate_multiplier(super::super::learning_rate::next_latent_scale())
            .shape(NEXTLAT_HIDDEN, GPT2_N_EMBD),
    );
    rows
}

fn schedule_slots(rows: &mut [HostPtrs]) {
    rows.sort_by_key(|slot| std::cmp::Reverse(estimated_polar_work(*slot)));
}

fn estimated_polar_work(slot: HostPtrs) -> u128 {
    let short = slot.rows.min(slot.cols) as u128;
    let long = slot.rows.max(slot.cols) as u128;
    short * short * long
}

fn pad_slots(rows: &mut Vec<HostPtrs>, padding: &AuroraPaddingBuffers) {
    while rows.len() % AURORA_MATRIX_PHASES != 0 {
        rows.push(padding.ptrs());
    }
}

fn append<F>(rows: &mut Vec<HostPtrs>, row_count: usize, col_count: usize, ptrs: F)
where
    F: Fn(usize) -> HostPtrs,
{
    for i in 0..GPT2_N_LAYER {
        rows.push(ptrs(i).shape(row_count, col_count));
    }
}

impl HostPtrs {
    fn descriptor(self) -> AuroraSlotDescriptor {
        AuroraSlotDescriptor {
            grad: self.grad,
            momentum: self.momentum,
            z_master: self.z_master,
            x_master: self.x_master,
            bytes: self.bytes,
            scales: self.scales,
            global_scale: self.global_scale,
            rows: self.rows,
            cols: self.cols,
            learning_rate_multiplier: self.learning_rate_multiplier,
        }
    }

    fn shape(mut self, rows: usize, cols: usize) -> Self {
        self.rows = rows as u32;
        self.cols = cols as u32;
        self
    }

    fn learning_rate_multiplier(mut self, value: f32) -> Self {
        self.learning_rate_multiplier = value;
        self
    }
}
