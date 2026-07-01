use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;

use super::device::{clone_device, decode_master};
use crate::{training::device_buffer::zero, upload::UploadedNvfp4};

pub(in crate::training) struct AdamState {
    pub(in crate::training) z_master: DeviceBuffer<f32>,
    pub(in crate::training) x_master: DeviceBuffer<f32>,
    pub(in crate::training) first: DeviceBuffer<f32>,
    pub(in crate::training) second: DeviceBuffer<f32>,
}

impl AdamState {
    pub(super) fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        tensor: &UploadedNvfp4,
    ) -> Result<Self, DriverError> {
        let master = decode_master(stream, decode, tensor)?;
        Ok(Self {
            z_master: clone_device(stream, &master)?,
            x_master: master,
            first: zero(stream, tensor.len)?,
            second: zero(stream, tensor.len)?,
        })
    }
}

pub(in crate::training) struct AuroraState {
    pub(in crate::training) z_master: DeviceBuffer<f32>,
    pub(in crate::training) x_master: DeviceBuffer<f32>,
    pub(in crate::training) momentum: DeviceBuffer<f32>,
}

impl AuroraState {
    pub(super) fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        tensor: &UploadedNvfp4,
    ) -> Result<Self, DriverError> {
        let master = decode_master(stream, decode, tensor)?;
        Ok(Self {
            z_master: clone_device(stream, &master)?,
            x_master: master,
            momentum: zero(stream, tensor.len)?,
        })
    }
}
