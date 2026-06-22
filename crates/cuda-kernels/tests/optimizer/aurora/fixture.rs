use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::optimizer::{AuroraMegaUpdateArgs, AuroraSlotDescriptor, OptimizerModule};

use crate::assertions::assert_update_matches;
use crate::common;
use crate::polar_reference::polar_first_iteration_scalar;

#[path = "fixture/buffers.rs"]
mod buffers;
use buffers::{Scratch, Slots, assert_quantized_slot_matches};

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
    let slot_descriptors = descriptors(&slots, row_count, col_count);
    let slot_descriptors = DeviceBuffer::from_host(&stream, &slot_descriptors)?;

    module.aurora_mega_update(AuroraMegaUpdateArgs {
        stream: &stream,
        slots: &slot_descriptors,
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

fn descriptors(slots: &Slots, rows: usize, cols: usize) -> Vec<AuroraSlotDescriptor> {
    (0..SLOT_COUNT)
        .map(|slot| AuroraSlotDescriptor {
            grad: slots.grads[slot].cu_deviceptr(),
            momentum: slots.momentums[slot].cu_deviceptr(),
            z_master: slots.z_masters[slot].cu_deviceptr(),
            x_master: slots.x_masters[slot].cu_deviceptr(),
            bytes: slots.bytes[slot].cu_deviceptr(),
            scales: slots.scales[slot].cu_deviceptr(),
            global_scale: slots.global_scales[slot].cu_deviceptr(),
            rows: rows as u32,
            cols: cols as u32,
            learning_rate_multiplier: 1.0,
        })
        .collect()
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
