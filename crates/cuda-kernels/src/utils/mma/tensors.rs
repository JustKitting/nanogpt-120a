use cuda_core::DeviceBuffer;

#[derive(Clone, Copy)]
pub struct Nvfp4FourSixMmaWeightTensor<'a> {
    pub bytes: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub global_scale: &'a DeviceBuffer<f32>,
}

pub struct Nvfp4DeviceScaleMmaWeightTensor<'a> {
    pub bytes: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub global_scale: &'a DeviceBuffer<f32>,
}
