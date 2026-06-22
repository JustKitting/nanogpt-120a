use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::optimizer::{AuroraMegaUpdateArgs, OptimizerModule};

use crate::assertions::assert_update_matches;
use crate::common;
use crate::polar_reference::polar_first_iteration_scalar;

#[path = "fixture/buffers.rs"]
mod buffers;
use buffers::{Scratch, Slots, assert_quantized_slot_matches, ptr_buffer};

const SLOT_COUNT: usize = rust_kernels_cuda::optimizer::AURORA_MATRIX_PHASES;

pub fn run_first_iteration_case(row_count: usize, col_count: usize) -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        OptimizerModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;
    let len = row_count * col_count;
    let gram_dim = row_count.min(col_count);
    let mut slots = Slots::new(&stream, len)?;
    let mut scratch = Scratch::new(&stream, len, gram_dim)?;
    let rows = DeviceBuffer::from_host(&stream, &[row_count as u32; SLOT_COUNT])?;
    let cols = DeviceBuffer::from_host(&stream, &[col_count as u32; SLOT_COUNT])?;
    let learning_rate_multipliers = DeviceBuffer::from_host(&stream, &[1.0_f32; SLOT_COUNT])?;
    let grad_ptrs = ptr_buffer(&stream, &slots.grads)?;
    let momentum_ptrs = ptr_buffer(&stream, &slots.momentums)?;
    let z_ptrs = ptr_buffer(&stream, &slots.z_masters)?;
    let x_ptrs = ptr_buffer(&stream, &slots.x_masters)?;
    let byte_ptrs = ptr_buffer(&stream, &slots.bytes)?;
    let scale_ptrs = ptr_buffer(&stream, &slots.scales)?;
    let global_scale_ptrs = ptr_buffer(&stream, &slots.global_scales)?;

    module.aurora_mega_update(AuroraMegaUpdateArgs {
        stream: &stream,
        grad_ptrs: &grad_ptrs,
        momentum_ptrs: &momentum_ptrs,
        z_master_ptrs: &z_ptrs,
        x_master_ptrs: &x_ptrs,
        byte_ptrs: &byte_ptrs,
        scale_ptrs: &scale_ptrs,
        global_scale_ptrs: &global_scale_ptrs,
        rows: &rows,
        cols: &cols,
        learning_rate_multipliers: &learning_rate_multipliers,
        oriented: &mut scratch.oriented,
        polar_next: &mut scratch.polar_next,
        polar_x: &mut scratch.polar_x,
        polar_gram: &mut scratch.polar_gram,
        polar_ax: &mut scratch.polar_ax,
        polar_chunks: &mut scratch.polar_chunks,
        slot_count: SLOT_COUNT as u32,
        max_len: len as u32,
        max_ax_len: len as u32,
        max_dim: gram_dim as u32,
        mu: MU,
        learning_rate: LEARNING_RATE,
        weight_decay: WEIGHT_DECAY,
        average_coefficient: 1.0,
        iterations: 1,
    })?;

    let expected = expected_update(row_count, col_count);
    assert_update_matches(
        &stream,
        slots.x_masters.remove(0),
        slots.z_masters.remove(0),
        expected,
    )?;
    assert_quantized_slot_matches(&stream, slots, expected)
}

const GRAD_VALUE: f32 = 0.5;
const LEARNING_RATE: f32 = 0.25;
const MU: f32 = 0.95;
const WEIGHT_DECAY: f32 = 0.1;

fn expected_update(rows: usize, cols: usize) -> f32 {
    let nesterov = (1.0 - MU) * (1.0 + MU) * GRAD_VALUE;
    let update = polar_first_iteration_scalar(nesterov, rows, cols);
    let scale = 0.2 * (rows.max(cols) as f32).sqrt();
    (1.0 - LEARNING_RATE * WEIGHT_DECAY) - LEARNING_RATE * update * scale
}
