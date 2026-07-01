use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::{
    LinearBackwardMsEdenScratch, LinearBackwardMsEdenScratchBuffers,
};

use super::attention::{AttentionCProjScratch, AttentionQkvScratch};
use super::final_head::FinalHeadBackwardScratch;
use crate::GPT2_TOKEN_ROWS;

pub struct LinearScratch {
    pub error_t: DeviceBuffer<f32>,
    pub weight_t: DeviceBuffer<f32>,
    pub input_t: DeviceBuffer<f32>,
    linear: LinearBackwardMsEdenScratchBuffers,
}

impl LinearScratch {
    pub fn new(
        stream: &CudaStream,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            error_t: DeviceBuffer::zeroed(stream, output_dim * GPT2_TOKEN_ROWS)?,
            weight_t: DeviceBuffer::zeroed(stream, output_dim * input_dim)?,
            input_t: DeviceBuffer::zeroed(stream, input_dim * GPT2_TOKEN_ROWS)?,
            linear: LinearBackwardMsEdenScratchBuffers::new(
                stream,
                GPT2_TOKEN_ROWS,
                input_dim,
                output_dim,
            )?,
        })
    }

    pub fn c_proj(&mut self) -> AttentionCProjScratch<'_> {
        let (error_t, weight_t, input_t, linear) = self.parts();
        AttentionCProjScratch {
            error_t,
            weight_t,
            input_t,
            linear,
        }
    }

    pub fn qkv(&mut self) -> AttentionQkvScratch<'_> {
        self.c_proj()
    }

    pub fn final_head(&mut self) -> FinalHeadBackwardScratch<'_> {
        let (error_t, weight_t, input_t, linear) = self.parts();
        FinalHeadBackwardScratch {
            dlogits_t: error_t,
            lm_head_weight_t: weight_t,
            final_normalized_t: input_t,
            linear,
        }
    }

    pub fn parts(
        &mut self,
    ) -> (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        LinearBackwardMsEdenScratch<'_>,
    ) {
        let Self {
            error_t,
            weight_t,
            input_t,
            linear,
        } = self;
        (error_t, weight_t, input_t, linear.as_args())
    }
}
