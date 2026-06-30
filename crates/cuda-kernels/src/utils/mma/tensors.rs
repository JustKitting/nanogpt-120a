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

impl<'a> From<Nvfp4FourSixMmaWeightTensor<'a>> for Nvfp4DeviceScaleMmaWeightTensor<'a> {
    fn from(tensor: Nvfp4FourSixMmaWeightTensor<'a>) -> Self {
        Self {
            bytes: tensor.bytes,
            scales: tensor.scales,
            global_scale: tensor.global_scale,
        }
    }
}
