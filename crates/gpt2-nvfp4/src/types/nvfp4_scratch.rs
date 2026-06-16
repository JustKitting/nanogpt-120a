use cuda_core::DeviceBuffer;

pub struct HiddenStateNvfp4<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scales: &'a mut DeviceBuffer<f32>,
}

impl<'a> HiddenStateNvfp4<'a> {
    pub fn reborrow(&mut self) -> HiddenStateNvfp4<'_> {
        HiddenStateNvfp4 {
            bytes: &mut *self.bytes,
            scales: &mut *self.scales,
            global_scales: &mut *self.global_scales,
        }
    }
}
