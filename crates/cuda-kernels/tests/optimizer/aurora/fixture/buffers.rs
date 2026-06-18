use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
use rust_kernels_cuda::optimizer::{AURORA_COOPERATIVE_BLOCKS, AURORA_MATRIX_PHASES};

use super::{GRAD_VALUE, SLOT_COUNT};

pub struct Slots {
    pub grads: Vec<DeviceBuffer<f32>>,
    pub momentums: Vec<DeviceBuffer<f32>>,
    pub z_masters: Vec<DeviceBuffer<f32>>,
    pub x_masters: Vec<DeviceBuffer<f32>>,
    pub bytes: Vec<DeviceBuffer<u8>>,
    pub scales: Vec<DeviceBuffer<u8>>,
    pub global_scales: Vec<DeviceBuffer<f32>>,
}

pub struct Scratch {
    pub oriented: DeviceBuffer<f32>,
    pub polar_next: DeviceBuffer<f32>,
    pub polar_x: DeviceBuffer<f32>,
    pub polar_gram: DeviceBuffer<f32>,
    pub polar_ax: DeviceBuffer<f32>,
    pub polar_chunks: DeviceBuffer<f32>,
}

impl Slots {
    pub fn new(stream: &CudaStream, len: usize) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            grads: slot_buffers(stream, GRAD_VALUE, len)?,
            momentums: zero_slot_buffers::<f32>(stream, len)?,
            z_masters: slot_buffers(stream, 1.0, len)?,
            x_masters: slot_buffers(stream, 1.0, len)?,
            bytes: zero_slot_buffers::<u8>(stream, len / 2)?,
            scales: zero_slot_buffers::<u8>(stream, len / 16)?,
            global_scales: zero_slot_buffers::<f32>(stream, 1)?,
        })
    }
}

impl Scratch {
    pub fn new(stream: &CudaStream, len: usize, gram_dim: usize) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            oriented: DeviceBuffer::<f32>::zeroed(stream, grouped(len))?,
            polar_next: DeviceBuffer::<f32>::zeroed(stream, grouped(len))?,
            polar_x: DeviceBuffer::<f32>::zeroed(stream, grouped(len))?,
            polar_gram: DeviceBuffer::<f32>::zeroed(stream, grouped(gram_dim * gram_dim))?,
            polar_ax: DeviceBuffer::<f32>::zeroed(stream, grouped(len))?,
            polar_chunks: DeviceBuffer::<f32>::zeroed(stream, grouped(AURORA_COOPERATIVE_BLOCKS))?,
        })
    }
}

const fn grouped(len: usize) -> usize {
    len * (SLOT_COUNT / AURORA_MATRIX_PHASES)
}

pub fn ptr_buffer<T>(
    stream: &CudaStream,
    buffers: &[DeviceBuffer<T>],
) -> Result<DeviceBuffer<u64>, Box<dyn Error>> {
    let ptrs: Vec<u64> = buffers.iter().map(DeviceBuffer::cu_deviceptr).collect();
    Ok(DeviceBuffer::from_host(stream, &ptrs)?)
}

pub fn assert_quantized_slot_matches(
    stream: &CudaStream,
    mut slots: Slots,
    expected: f32,
) -> Result<(), Box<dyn Error>> {
    let bytes = slots.bytes.remove(0).to_host_vec(stream)?;
    let scales = slots.scales.remove(0).to_host_vec(stream)?;
    let global_scale = slots.global_scales.remove(0).to_host_vec(stream)?;
    assert!(bytes.iter().any(|byte| *byte != 0));
    assert!(scales.iter().any(|scale| *scale != 0));
    assert!((global_scale[0] - expected.abs() / (256.0 * 6.0)).abs() <= 1.0e-8);
    Ok(())
}

fn slot_buffers(
    stream: &CudaStream,
    value: f32,
    len: usize,
) -> Result<Vec<DeviceBuffer<f32>>, Box<dyn Error>> {
    let mut buffers = Vec::with_capacity(SLOT_COUNT);
    for _ in 0..SLOT_COUNT {
        buffers.push(DeviceBuffer::from_host(stream, &vec![value; len])?);
    }
    Ok(buffers)
}

fn zero_slot_buffers<T>(
    stream: &CudaStream,
    len: usize,
) -> Result<Vec<DeviceBuffer<T>>, Box<dyn Error>>
where
    T: DeviceCopy,
{
    let mut buffers = Vec::with_capacity(SLOT_COUNT);
    for _ in 0..SLOT_COUNT {
        buffers.push(DeviceBuffer::<T>::zeroed(stream, len)?);
    }
    Ok(buffers)
}
