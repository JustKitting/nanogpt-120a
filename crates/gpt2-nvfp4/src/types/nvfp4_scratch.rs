use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

pub struct RowwiseNvfp4Scratch<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scales: &'a mut DeviceBuffer<f32>,
}

impl<'a> RowwiseNvfp4Scratch<'a> {
    pub fn device(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor {
            bytes: &*self.bytes,
            scales: &*self.scales,
            global_scales: &*self.global_scales,
        }
    }

    pub fn reborrow(&mut self) -> RowwiseNvfp4Scratch<'_> {
        RowwiseNvfp4Scratch {
            bytes: &mut *self.bytes,
            scales: &mut *self.scales,
            global_scales: &mut *self.global_scales,
        }
    }
}

pub type HiddenStateNvfp4<'a> = RowwiseNvfp4Scratch<'a>;
pub type MlpActivationNvfp4<'a> = RowwiseNvfp4Scratch<'a>;
