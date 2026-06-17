use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{AttentionCProjScratch, FinalHeadBackwardScratch, GPT2_CONTEXT_LEN};
use rust_kernels_cuda::linear_backward::LinearBackwardMsEdenScratch;
use rust_kernels_cuda::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;

use super::operand_scratch::OperandScratch;

pub struct LinearScratch {
    pub error_t: DeviceBuffer<f32>,
    pub weight_t: DeviceBuffer<f32>,
    pub input_t: DeviceBuffer<f32>,
    e: OperandScratch,
    weight_t_h: OperandScratch,
    e_t: OperandScratch,
    input_t_h: OperandScratch,
}

impl LinearScratch {
    pub fn new(
        stream: &CudaStream,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DriverError> {
        let output_k = nvfp4_tc_matmul_padded_k(output_dim as u32) as usize;
        let token_k = nvfp4_tc_matmul_padded_k(GPT2_CONTEXT_LEN as u32) as usize;

        Ok(Self {
            error_t: DeviceBuffer::zeroed(stream, output_dim * GPT2_CONTEXT_LEN)?,
            weight_t: DeviceBuffer::zeroed(stream, output_dim * input_dim)?,
            input_t: DeviceBuffer::zeroed(stream, input_dim * GPT2_CONTEXT_LEN)?,
            e: OperandScratch::new(stream, GPT2_CONTEXT_LEN * output_k, GPT2_CONTEXT_LEN)?,
            weight_t_h: OperandScratch::new(stream, input_dim * output_k, input_dim)?,
            e_t: OperandScratch::new(stream, output_dim * token_k, output_dim)?,
            input_t_h: OperandScratch::new(stream, input_dim * token_k, input_dim)?,
        })
    }

    pub fn attention(&mut self) -> AttentionCProjScratch<'_> {
        let Self {
            error_t,
            weight_t,
            input_t,
            e,
            weight_t_h,
            e_t,
            input_t_h,
        } = self;
        AttentionCProjScratch {
            error_t,
            weight_t,
            input_t,
            linear: ms_eden(e, weight_t_h, e_t, input_t_h),
        }
    }

    pub fn final_head(&mut self) -> FinalHeadBackwardScratch<'_> {
        let Self {
            error_t,
            weight_t,
            input_t,
            e,
            weight_t_h,
            e_t,
            input_t_h,
        } = self;
        FinalHeadBackwardScratch {
            dlogits_t: error_t,
            lm_head_weight_t: weight_t,
            final_normalized_t: input_t,
            linear: ms_eden(e, weight_t_h, e_t, input_t_h),
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
            ms_eden(e, weight_t_h, e_t, input_t_h),
        )
    }
}

fn ms_eden<'a>(
    e: &'a mut OperandScratch,
    weight_t_h: &'a mut OperandScratch,
    e_t: &'a mut OperandScratch,
    input_t_h: &'a mut OperandScratch,
) -> LinearBackwardMsEdenScratch<'a> {
    LinearBackwardMsEdenScratch {
        e_h: e.operand(),
        weight_t_h: weight_t_h.operand(),
        e_t_h: e_t.operand(),
        input_t_h: input_t_h.operand(),
    }
}
