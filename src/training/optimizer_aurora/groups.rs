use super::super::grads::BackwardBuffers;
use super::super::optimizer_state::OptimizerStateBuffers;
use crate::upload::UploadedModel;
use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD, GPT2_N_LAYER, GPT2_QKV};

mod ptrs;

pub(in crate::training) struct AuroraPointerTables {
    pub(in crate::training) all: AuroraGroupTable,
    pub(in crate::training) slot_count: usize,
}

pub(in crate::training) struct AuroraGroupTable {
    pub(super) grad: DeviceBuffer<u64>,
    pub(super) momentum: DeviceBuffer<u64>,
    pub(super) z_master: DeviceBuffer<u64>,
    pub(super) x_master: DeviceBuffer<u64>,
    pub(super) bytes: DeviceBuffer<u64>,
    pub(super) scales: DeviceBuffer<u64>,
    pub(super) global_scale: DeviceBuffer<u64>,
    pub(super) rows: DeviceBuffer<u32>,
    pub(super) cols: DeviceBuffer<u32>,
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
}

impl AuroraPointerTables {
    pub(in crate::training) fn new(
        stream: &CudaStream,
        uploaded: &UploadedModel,
        grads: &BackwardBuffers,
        state: &OptimizerStateBuffers,
    ) -> Result<Self, DriverError> {
        let rows = all_slots(uploaded, grads, state);
        Ok(Self {
            all: AuroraGroupTable::new(stream, &rows)?,
            slot_count: rows.len(),
        })
    }
}

impl AuroraGroupTable {
    fn new(stream: &CudaStream, rows: &[HostPtrs]) -> Result<Self, DriverError> {
        Ok(Self {
            grad: upload(stream, &rows, |p| p.grad)?,
            momentum: upload(stream, &rows, |p| p.momentum)?,
            z_master: upload(stream, &rows, |p| p.z_master)?,
            x_master: upload(stream, &rows, |p| p.x_master)?,
            bytes: upload(stream, &rows, |p| p.bytes)?,
            scales: upload(stream, &rows, |p| p.scales)?,
            global_scale: upload(stream, &rows, |p| p.global_scale)?,
            rows: upload(stream, &rows, |p| p.rows)?,
            cols: upload(stream, &rows, |p| p.cols)?,
        })
    }
}

fn upload<T, F>(
    stream: &CudaStream,
    rows: &[HostPtrs],
    f: F,
) -> Result<DeviceBuffer<T>, DriverError>
where
    T: DeviceCopy,
    F: Fn(HostPtrs) -> T,
{
    let values: Vec<T> = rows.iter().copied().map(f).collect();
    DeviceBuffer::from_host(stream, &values)
}

fn all_slots(
    uploaded: &UploadedModel,
    grads: &BackwardBuffers,
    state: &OptimizerStateBuffers,
) -> Vec<HostPtrs> {
    let mut rows = Vec::with_capacity(GPT2_N_LAYER * 4);
    append(&mut rows, GPT2_N_EMBD, GPT2_QKV, |i| {
        ptrs::qkv(uploaded, grads, state, i)
    });
    append(&mut rows, GPT2_N_EMBD, GPT2_N_EMBD, |i| {
        ptrs::c_proj(uploaded, grads, state, i)
    });
    append(&mut rows, GPT2_N_EMBD, GPT2_MLP, |i| {
        ptrs::mlp_up(uploaded, grads, state, i)
    });
    append(&mut rows, GPT2_MLP, GPT2_N_EMBD, |i| {
        ptrs::mlp_down(uploaded, grads, state, i)
    });
    rows
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
    fn shape(mut self, rows: usize, cols: usize) -> Self {
        self.rows = rows as u32;
        self.cols = cols as u32;
        self
    }
}
