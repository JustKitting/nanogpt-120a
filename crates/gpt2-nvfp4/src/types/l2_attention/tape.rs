use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use crate::types::RowwiseNvfp4Tape;

pub struct AttentionForwardTape<'scratch> {
    pub qkv_input_nvfp4: RowwiseNvfp4Tape<'scratch>,
    pub c_proj_input_nvfp4: RowwiseNvfp4Tape<'scratch>,
}

impl<'scratch> AttentionForwardTape<'scratch> {
    pub(crate) fn save_qkv_input(
        &mut self,
        stream: &CudaStream,
        input: Nvfp4RowwiseDeviceTensor<'_>,
    ) -> Result<(), DriverError> {
        self.qkv_input_nvfp4.save(stream, input)
    }

    pub(crate) fn save_c_proj_input(
        &mut self,
        stream: &CudaStream,
        input: Nvfp4RowwiseDeviceTensor<'_>,
    ) -> Result<(), DriverError> {
        self.c_proj_input_nvfp4.save(stream, input)
    }
}
