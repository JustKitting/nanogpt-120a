use cuda_core::DeviceBuffer;
use rust_kernels_cuda::optimizer::GRAD_CLIP_VALUES_PER_CHUNK;

use crate::training::grads::BackwardBuffers;
use crate::training::next_latent::NextLatGradBuffers;

mod scan;
mod views;

pub(in crate::training) use scan::first_non_finite_gradient;
use views::parameter_gradient_views;

#[derive(Clone, Copy)]
pub(super) struct HostGradPtr {
    pub(super) ptr: u64,
    pub(super) len: u32,
    pub(super) chunk_offset: u32,
}

pub(super) fn parameter_gradients(
    grads: &BackwardBuffers,
    next_latent: &NextLatGradBuffers,
) -> Vec<HostGradPtr> {
    let views = parameter_gradient_views(grads, next_latent);
    let mut rows = Vec::new();
    for view in views {
        push(&mut rows, view.buffer, view.len);
    }
    rows
}

pub(super) fn gradient_chunk_count(rows: &[HostGradPtr]) -> u32 {
    rows.last()
        .map(|row| row.chunk_offset + chunks(row.len))
        .unwrap_or(0)
}

fn push(rows: &mut Vec<HostGradPtr>, buffer: &DeviceBuffer<f32>, len: usize) {
    let chunk_offset = gradient_chunk_count(rows);
    rows.push(HostGradPtr {
        ptr: buffer.cu_deviceptr(),
        len: len as u32,
        chunk_offset,
    });
}

fn chunks(len: u32) -> u32 {
    len.div_ceil(GRAD_CLIP_VALUES_PER_CHUNK as u32)
}
