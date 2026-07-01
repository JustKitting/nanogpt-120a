use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{AttentionCProjScratch, FinalHeadBackwardScratch, GPT2_TOKEN_ROWS};
use rust_kernels_cuda::linear_backward::{LinearBackwardMsEdenScratch, MsEdenOperandScratchBuffer};
use rust_kernels_cuda::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;

use super::device_buffer::zero;

pub struct LinearScratch {
    pub error_t: DeviceBuffer<f32>,
    pub weight_t: DeviceBuffer<f32>,
    pub input_t: DeviceBuffer<f32>,
    e: MsEdenOperandScratchBuffer,
    weight_t_h: MsEdenOperandScratchBuffer,
    e_t: MsEdenOperandScratchBuffer,
    input_t_h: MsEdenOperandScratchBuffer,
}

impl LinearScratch {
    pub fn new(
        stream: &CudaStream,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DriverError> {
        let output_k = nvfp4_tc_matmul_padded_k(output_dim as u32) as usize;
        let token_k = nvfp4_tc_matmul_padded_k(GPT2_TOKEN_ROWS as u32) as usize;

        Ok(Self {
            error_t: zero(stream, output_dim * GPT2_TOKEN_ROWS)?,
            weight_t: zero(stream, output_dim * input_dim)?,
            input_t: zero(stream, input_dim * GPT2_TOKEN_ROWS)?,
            e: MsEdenOperandScratchBuffer::new(
                stream,
                GPT2_TOKEN_ROWS * output_k,
                GPT2_TOKEN_ROWS,
            )?,
            weight_t_h: MsEdenOperandScratchBuffer::new(stream, input_dim * output_k, input_dim)?,
            e_t: MsEdenOperandScratchBuffer::new(stream, output_dim * token_k, output_dim)?,
            input_t_h: MsEdenOperandScratchBuffer::new(stream, input_dim * token_k, input_dim)?,
        })
    }

    pub fn attention(&mut self) -> AttentionCProjScratch<'_> {
        let (error_t, weight_t, input_t, linear) = self.parts();
        AttentionCProjScratch {
            error_t,
            weight_t,
            input_t,
            linear,
        }
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
            e,
            weight_t_h,
            e_t,
            input_t_h,
        } = self;
        (
            error_t,
            weight_t,
            input_t,
            LinearBackwardMsEdenScratch {
                e_h: e.as_arg(),
                weight_t_h: weight_t_h.as_arg(),
                e_t_h: e_t.as_arg(),
                input_t_h: input_t_h.as_arg(),
            },
        )
    }
}
