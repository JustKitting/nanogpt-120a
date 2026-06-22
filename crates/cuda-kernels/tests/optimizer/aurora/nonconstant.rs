use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::optimizer::{AURORA_MATRIX_PHASES, AuroraMegaUpdateArgs, OptimizerModule};

use crate::{common, polar_vector};

#[path = "nonconstant_buffers.rs"]
mod nonconstant_buffers;
use nonconstant_buffers::{Scratch, Slots, ptr_buffer};

const ROWS: usize = 32;
const COLS: usize = 64;
const LEN: usize = ROWS * COLS;
const SLOT_COUNT: usize = AURORA_MATRIX_PHASES;
const MU: f32 = 0.95;
const LEARNING_RATE: f32 = 0.25;
const WEIGHT_DECAY: f32 = 0.1;
const ITERATIONS: u32 = 5;

pub fn run_wide_case() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        OptimizerModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;
    let grad = gradient();
    let mut slots = Slots::new(&stream, &grad)?;
    let mut scratch = Scratch::new(&stream)?;
    let rows = DeviceBuffer::from_host(&stream, &[ROWS as u32; SLOT_COUNT])?;
    let cols = DeviceBuffer::from_host(&stream, &[COLS as u32; SLOT_COUNT])?;
    let learning_rate_multipliers = DeviceBuffer::from_host(&stream, &[1.0_f32; SLOT_COUNT])?;

    module.aurora_mega_update(AuroraMegaUpdateArgs {
        stream: &stream,
        grad_ptrs: &ptr_buffer(&stream, &slots.grads)?,
        momentum_ptrs: &ptr_buffer(&stream, &slots.momentums)?,
        z_master_ptrs: &ptr_buffer(&stream, &slots.z_masters)?,
        x_master_ptrs: &ptr_buffer(&stream, &slots.x_masters)?,
        byte_ptrs: &ptr_buffer(&stream, &slots.bytes)?,
        scale_ptrs: &ptr_buffer(&stream, &slots.scales)?,
        global_scale_ptrs: &ptr_buffer(&stream, &slots.global_scales)?,
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
        max_len: LEN as u32,
        max_ax_len: LEN as u32,
        max_dim: ROWS as u32,
        mu: MU,
        learning_rate: LEARNING_RATE,
        weight_decay: WEIGHT_DECAY,
        average_coefficient: 1.0,
        iterations: ITERATIONS,
    })?;

    let expected = polar_vector::first_iteration_update(
        &grad,
        ROWS,
        COLS,
        MU,
        LEARNING_RATE,
        WEIGHT_DECAY,
        ITERATIONS as usize,
    );
    assert_close(&slots.x_masters.remove(0).to_host_vec(&stream)?, &expected);
    assert_close(&slots.z_masters.remove(0).to_host_vec(&stream)?, &expected);
    Ok(())
}

fn gradient() -> Vec<f32> {
    (0..LEN)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.0003 + ((i / COLS) as f32) * 0.00001)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    let mut max_error = 0.0_f32;
    let mut max_index = 0;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        if error > max_error {
            max_error = error;
            max_index = index;
        }
    }
    assert!(
        max_error <= 2.0e-3,
        "max_error={max_error:.8e} index={max_index} actual={:.8e} expected={:.8e}",
        actual[max_index],
        expected[max_index],
    );
}
