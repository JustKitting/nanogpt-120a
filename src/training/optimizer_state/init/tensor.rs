use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;

use super::super::{
    AdamState, AuroraState,
    device::{clone_device, decode_master},
};
use crate::{training::device_buffer::zero, upload::UploadedNvfp4};

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
