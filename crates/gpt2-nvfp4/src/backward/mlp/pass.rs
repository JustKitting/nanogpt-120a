use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::LinearBackwardMsEdenScratch;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use super::args::MlpBackwardModules;
use super::linear::{MlpLinearBackwardCall, run_linear_backward};
use super::transforms::{decode_rowwise_t, decode_weight_t, transpose_f32};

pub(super) struct LinearPass<'a, 'scratch, 'out> {
    pub e: &'a DeviceBuffer<f32>,
    pub saved_input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub error_t: &'scratch mut DeviceBuffer<f32>,
    pub weight_t: &'scratch mut DeviceBuffer<f32>,
    pub input_t: &'scratch mut DeviceBuffer<f32>,
    pub linear_scratch: LinearBackwardMsEdenScratch<'scratch>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub(super) fn run_linear_pass(
    modules: &MlpBackwardModules<'_>,
    stream: &CudaStream,
    pass: LinearPass<'_, '_, '_>,
) -> Result<(), DriverError> {
    let row_count = pass.row_count;
    decode_weight_t(
        modules.decode,
        stream,
        pass.weight,
        pass.weight_t,
        pass.output_dim as usize,
        pass.input_dim as usize,
    )?;
    transpose_f32(
        modules.transpose,
        stream,
        pass.e,
        pass.error_t,
        row_count as usize,
        pass.output_dim as usize,
    )?;
    decode_rowwise_t(
        modules.decode,
        stream,
        pass.saved_input,
        pass.input_t,
        row_count as usize,
        pass.input_dim as usize,
    )?;
    run_linear_backward(
        modules.linear,
        modules.quant,
        stream,
        MlpLinearBackwardCall {
            e: pass.e,
            weight_t: pass.weight_t,
            e_t: pass.error_t,
            input_t: pass.input_t,
            scratch: pass.linear_scratch,
            dinput: pass.dinput,
            dweight: pass.dweight,
            dbias: pass.dbias,
            input_dim: pass.input_dim,
            output_dim: pass.output_dim,
            token_count: row_count,
            sign_seed: pass.sign_seed,
            scale_seed: pass.scale_seed,
        },
    )
}
