use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::optimizer::AURORA_COOPERATIVE_BLOCKS;

use super::{LEN, ROWS, SLOT_COUNT};

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
    pub fn new(stream: &CudaStream, grad: &[f32]) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            grads: slot_buffers(stream, grad)?,
            momentums: zero_slot_buffers::<f32>(stream, LEN)?,
            z_masters: fill_slot_buffers(stream, 1.0, LEN)?,
            x_masters: fill_slot_buffers(stream, 1.0, LEN)?,
            bytes: zero_slot_buffers::<u8>(stream, LEN / 2)?,
            scales: zero_slot_buffers::<u8>(stream, LEN / 16)?,
            global_scales: zero_slot_buffers::<f32>(stream, 1)?,
        })
    }
}

impl Scratch {
    pub fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            oriented: DeviceBuffer::<f32>::zeroed(stream, LEN)?,
            polar_next: DeviceBuffer::<f32>::zeroed(stream, LEN)?,
            polar_x: DeviceBuffer::<f32>::zeroed(stream, LEN)?,
            polar_gram: DeviceBuffer::<f32>::zeroed(stream, ROWS * ROWS)?,
            polar_ax: DeviceBuffer::<f32>::zeroed(stream, LEN)?,
            polar_chunks: DeviceBuffer::<f32>::zeroed(stream, AURORA_COOPERATIVE_BLOCKS)?,
        })
    }
}

pub fn ptr_buffer<T>(
    stream: &CudaStream,
    buffers: &[DeviceBuffer<T>],
) -> Result<DeviceBuffer<u64>, Box<dyn Error>> {
    let ptrs: Vec<u64> = buffers.iter().map(DeviceBuffer::cu_deviceptr).collect();
    Ok(DeviceBuffer::from_host(stream, &ptrs)?)
}

fn slot_buffers(
    stream: &CudaStream,
    values: &[f32],
) -> Result<Vec<DeviceBuffer<f32>>, Box<dyn Error>> {
    let mut out = Vec::with_capacity(SLOT_COUNT);
    for _ in 0..SLOT_COUNT {
        out.push(DeviceBuffer::from_host(stream, values)?);
    }
    Ok(out)
}

fn fill_slot_buffers(
    stream: &CudaStream,
    value: f32,
    len: usize,
) -> Result<Vec<DeviceBuffer<f32>>, Box<dyn Error>> {
    slot_buffers(stream, &vec![value; len])
}

fn zero_slot_buffers<T>(
    stream: &CudaStream,
    len: usize,
) -> Result<Vec<DeviceBuffer<T>>, Box<dyn Error>>
where
    T: cuda_core::DeviceCopy,
{
    let mut out = Vec::with_capacity(SLOT_COUNT);
    for _ in 0..SLOT_COUNT {
        out.push(DeviceBuffer::<T>::zeroed(stream, len)?);
    }
    Ok(out)
}
