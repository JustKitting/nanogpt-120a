use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::device_copy::copy_device;
use super::types::LayerNormTape;
use crate::types::LayerNormSaved;

impl<'a> LayerNormTape<'a> {
    pub(super) fn saved(&self, row_count: u32) -> LayerNormSaved<'_> {
        LayerNormSaved {
            row_count,
            residual: &*self.residual,
            normalized: &*self.normalized,
            mean: &*self.mean,
            inv_std: &*self.inv_std,
        }
    }

    pub(super) fn reborrow(&mut self) -> LayerNormTape<'_> {
        LayerNormTape {
            residual: &mut *self.residual,
            normalized: &mut *self.normalized,
            mean: &mut *self.mean,
            inv_std: &mut *self.inv_std,
        }
    }

    pub(crate) fn save(
        &mut self,
        stream: &CudaStream,
        residual: &DeviceBuffer<f32>,
        normalized: &DeviceBuffer<f32>,
        mean: &DeviceBuffer<f32>,
        inv_std: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, residual, self.residual)?;
        copy_device(stream, normalized, self.normalized)?;
        copy_device(stream, mean, self.mean)?;
        copy_device(stream, inv_std, self.inv_std)
    }
}
