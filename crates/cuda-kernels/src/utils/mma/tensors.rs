use cuda_core::DeviceBuffer;

#[derive(Clone, Copy)]
pub struct Nvfp4FourSixMmaWeightTensor<'a> {
    pub bytes: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub global_scale: &'a DeviceBuffer<f32>,
}

impl<'a> Nvfp4FourSixMmaWeightTensor<'a> {
    pub fn new(
        bytes: &'a DeviceBuffer<u8>,
        scales: &'a DeviceBuffer<u8>,
        global_scale: &'a DeviceBuffer<f32>,
    ) -> Self {
        Self {
            bytes,
            scales,
            global_scale,
        }
    }
}

pub struct Nvfp4DeviceScaleMmaWeightTensor<'a> {
    pub bytes: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub global_scale: &'a DeviceBuffer<f32>,
}

impl<'a> Nvfp4DeviceScaleMmaWeightTensor<'a> {
    pub fn new(
        bytes: &'a DeviceBuffer<u8>,
        scales: &'a DeviceBuffer<u8>,
        global_scale: &'a DeviceBuffer<f32>,
    ) -> Self {
        Self {
            bytes,
            scales,
            global_scale,
        }
    }
}

impl<'a> From<Nvfp4FourSixMmaWeightTensor<'a>> for Nvfp4DeviceScaleMmaWeightTensor<'a> {
    fn from(tensor: Nvfp4FourSixMmaWeightTensor<'a>) -> Self {
        Self::new(tensor.bytes, tensor.scales, tensor.global_scale)
    }
}
