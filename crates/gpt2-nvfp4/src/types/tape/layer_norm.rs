use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::device_copy::copy_device;
use super::types::LayerNormTape;
use crate::types::LayerNormSaved;

impl<'a> LayerNormTape<'a> {
    pub(super) fn saved(&self) -> LayerNormSaved<'_> {
        LayerNormSaved {
            residual: &*self.residual,
            normalized: &*self.normalized,
        }
    }

    pub(super) fn reborrow(&mut self) -> LayerNormTape<'_> {
        LayerNormTape {
            residual: &mut *self.residual,
            normalized: &mut *self.normalized,
        }
    }

    pub(crate) fn save(
        &mut self,
        stream: &CudaStream,
        residual: &DeviceBuffer<f32>,
        normalized: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, residual, self.residual)?;
        copy_device(stream, normalized, self.normalized)
    }
}
