use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD, NEXTLAT_HIDDEN, NEXTLAT_INPUT};
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
        slots: &args.table.slots,
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
    max2(GPT2_MLP * GPT2_N_EMBD, NEXTLAT_INPUT * NEXTLAT_HIDDEN)
}

const fn max_polar_ax_len() -> usize {
    max2(GPT2_N_EMBD * GPT2_N_EMBD, NEXTLAT_HIDDEN * NEXTLAT_HIDDEN)
}

const fn max_matrix_dim() -> usize {
    max2(GPT2_N_EMBD, NEXTLAT_HIDDEN)
}

const fn max2(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}
