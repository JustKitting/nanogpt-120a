use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::types::LayerNormTape;
use crate::types::LayerNormSaved;

impl<'a> LayerNormTape<'a> {
    pub(super) fn saved(&self, row_count: u32) -> LayerNormSaved<'_> {
        LayerNormSaved {
            row_count,
            residual: &*self.residual,
            mean: &*self.mean,
            inv_std: &*self.inv_std,
        }
    }

    pub(super) fn reborrow(&mut self) -> LayerNormTape<'_> {
        LayerNormTape {
            residual: &mut *self.residual,
            mean: &mut *self.mean,
            inv_std: &mut *self.inv_std,
        }
    }

    pub(crate) fn save_stats(
        &mut self,
        stream: &CudaStream,
        mean: &DeviceBuffer<f32>,
        inv_std: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        super::device_copy::copy_device(stream, mean, self.mean)?;
        super::device_copy::copy_device(stream, inv_std, self.inv_std)
    }
}
