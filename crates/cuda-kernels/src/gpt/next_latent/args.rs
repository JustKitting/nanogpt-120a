use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};

use crate::mma::{Nvfp4FourSixMmaWeightTensor, Nvfp4ProjectionParams};
use crate::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NextLatShape {
    pub row_count: u32,
    pub embedding_dim: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub lambda: f32,
}

unsafe impl DeviceCopy for NextLatShape {}

pub struct NextLatConcatArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub next_token_embeddings: &'a DeviceBuffer<f32>,
    pub current_states: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
}

pub struct NextLatConcatBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub d_concat: &'a DeviceBuffer<f32>,
    pub d_predicted: &'a DeviceBuffer<f32>,
    pub d_next_token_embeddings: &'out mut DeviceBuffer<f32>,
    pub d_current_states: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
}

pub struct NextLatSmoothL1Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub predicted_next_states: &'a DeviceBuffer<f32>,
    pub target_states: &'a DeviceBuffer<f32>,
    pub losses: &'out mut DeviceBuffer<f32>,
    pub d_predicted_next_states: &'out mut DeviceBuffer<f32>,
    pub batch_size: u32,
    pub seq_len: u32,
    pub embedding_dim: u32,
    pub lambda: f32,
}

pub struct NextLatProjectionArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
}

pub struct NextLatGeluArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub len: u32,
}

pub struct NextLatGeluBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub d_out: &'a DeviceBuffer<f32>,
    pub d_input: &'out mut DeviceBuffer<f32>,
    pub len: u32,
}

pub struct NextLatResidualAddArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub delta: &'a DeviceBuffer<f32>,
    pub residual: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub len: u32,
}

pub fn projection_params(args: &NextLatProjectionArgs<'_, '_>) -> Nvfp4ProjectionParams {
    Nvfp4ProjectionParams {
        token_count: args.token_count,
        input_dim: args.input_dim,
        output_dim: args.output_dim,
        weight_global_scale: 1.0,
        bias_global_scale: 1.0,
        residual_add: 0,
        activation: 0,
    }
}
