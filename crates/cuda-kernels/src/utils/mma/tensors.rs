use cuda_core::DeviceBuffer;

#[derive(Clone, Copy)]
pub struct Nvfp4FourSixMmaWeightTensor<'a> {
    pub bytes: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub global_scale: f32,
}
