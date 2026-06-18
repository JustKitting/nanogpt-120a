use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD};
use rust_kernels_cuda::optimizer::{AuroraMegaUpdateArgs, OptimizerModule};

use super::{AURORA_WEIGHT_DECAY, AuroraGroupTable, MU, POLAR_ITERATIONS, aurora_learning_rate};
use crate::training::optimizer_tc_scratch::AuroraScratchBuffers;

pub(in crate::training) struct AuroraMegaArgs<'a> {
    pub(in crate::training) stream: &'a CudaStream,
    pub(in crate::training) optimizer: &'a OptimizerModule,
    pub(in crate::training) table: &'a AuroraGroupTable,
    pub(in crate::training) scratch: &'a mut AuroraScratchBuffers,
    pub(in crate::training) slot_count: usize,
    pub(in crate::training) step: u32,
    pub(in crate::training) average_coefficient: f32,
}

pub(in crate::training) fn apply_aurora_mega(args: AuroraMegaArgs<'_>) -> Result<(), DriverError> {
    args.optimizer.aurora_mega_update(AuroraMegaUpdateArgs {
        stream: args.stream,
        grad_ptrs: &args.table.grad,
        momentum_ptrs: &args.table.momentum,
        z_master_ptrs: &args.table.z_master,
        x_master_ptrs: &args.table.x_master,
        byte_ptrs: &args.table.bytes,
        scale_ptrs: &args.table.scales,
        global_scale_ptrs: &args.table.global_scale,
        rows: &args.table.rows,
        cols: &args.table.cols,
        oriented: &mut args.scratch.oriented,
        polar_next: &mut args.scratch.polar_next,
        polar_x: &mut args.scratch.polar_x,
        polar_gram: &mut args.scratch.polar_gram,
        polar_ax: &mut args.scratch.polar_ax,
        polar_chunks: &mut args.scratch.polar_chunks,
        slot_count: args.slot_count as u32,
        max_len: max_matrix_len() as u32,
        max_ax_len: max_polar_ax_len() as u32,
        max_dim: max_matrix_dim() as u32,
        mu: MU,
        learning_rate: aurora_learning_rate(args.step),
        weight_decay: AURORA_WEIGHT_DECAY,
        average_coefficient: args.average_coefficient,
        iterations: POLAR_ITERATIONS,
    })
}

const fn max_matrix_len() -> usize {
    GPT2_MLP * GPT2_N_EMBD
}

const fn max_polar_ax_len() -> usize {
    GPT2_N_EMBD * GPT2_N_EMBD
}

const fn max_matrix_dim() -> usize {
    GPT2_N_EMBD
}
