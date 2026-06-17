use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use crate::types::RowwiseNvfp4Tape;

pub struct MlpForwardTape<'scratch> {
    pub up_input_nvfp4: RowwiseNvfp4Tape<'scratch>,
    pub down_input_nvfp4: RowwiseNvfp4Tape<'scratch>,
}

impl<'scratch> MlpForwardTape<'scratch> {
    pub(crate) fn save_up_input(
        &mut self,
        stream: &CudaStream,
        input: Nvfp4RowwiseDeviceTensor<'_>,
    ) -> Result<(), DriverError> {
        self.up_input_nvfp4.save(stream, input)
    }

    pub(crate) fn save_down_input(
        &mut self,
        stream: &CudaStream,
        input: Nvfp4RowwiseDeviceTensor<'_>,
    ) -> Result<(), DriverError> {
        self.down_input_nvfp4.save(stream, input)
    }
}
