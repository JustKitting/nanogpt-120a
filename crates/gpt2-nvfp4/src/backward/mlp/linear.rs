use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::{
    LinearBackwardModule, LinearBackwardMsEdenArgs, LinearBackwardMsEdenScratch,
};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

pub(super) struct MlpLinearBackwardCall<'a, 'scratch, 'out> {
    pub e: &'a DeviceBuffer<f32>,
    pub weight_t: &'scratch DeviceBuffer<f32>,
    pub e_t: &'scratch DeviceBuffer<f32>,
    pub input_t: &'scratch DeviceBuffer<f32>,
    pub scratch: LinearBackwardMsEdenScratch<'scratch>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: &'out mut DeviceBuffer<f32>,
    pub input_dim: u32,
    pub output_dim: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub(super) fn run_linear_backward(
    module: &LinearBackwardModule,
    quant: &Nvfp4QuantModule,
    stream: &CudaStream,
    call: MlpLinearBackwardCall<'_, '_, '_>,
) -> Result<(), DriverError> {
    module.backward_ms_eden(LinearBackwardMsEdenArgs {
        stream,
        quant_module: quant,
        e: call.e,
        weight_t: call.weight_t,
        e_t: call.e_t,
        input_t: call.input_t,
        scratch: call.scratch,
        dinput: call.dinput,
        dweight: call.dweight,
        dbias: Some(call.dbias),
        token_count: crate::GPT2_CONTEXT_LEN as u32,
        input_dim: call.input_dim,
        output_dim: call.output_dim,
        sign_seed: call.sign_seed,
        scale_seed: call.scale_seed,
    })
}
