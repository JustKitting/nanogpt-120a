use cuda_core::{CudaStream, DeviceBuffer};

pub struct Nvfp4WeightUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scale: f32,
    pub aurora_update: &'a DeviceBuffer<f32>,
    pub fp32_workspace: &'a mut DeviceBuffer<f32>,
    pub amax: &'a mut DeviceBuffer<f32>,
    pub next_global_scale: &'a mut DeviceBuffer<f32>,
    pub len: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
}
