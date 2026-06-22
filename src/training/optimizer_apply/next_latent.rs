use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::optimizer::OptimizerModule;

use crate::training::next_latent::NextLatGradBuffers;
use crate::training::optimizer::OptimizerScratch;
use crate::training::optimizer_state::NextLatState;
use crate::upload::UploadedNextLat;

use super::adam::{next_latent_adam_learning_rate, update_adam_tensor_with_learning_rate};

pub(super) struct NextLatUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub optimizer: &'a OptimizerModule,
    pub weights: &'a mut UploadedNextLat,
    pub grads: &'a NextLatGradBuffers,
    pub scratch: &'a mut OptimizerScratch,
    pub state: &'a mut NextLatState,
    pub step: u32,
    pub average_coefficient: f32,
}

pub(super) fn update_next_latent(args: NextLatUpdateArgs<'_>) -> Result<(), DriverError> {
    let step = args.step;
    let average = args.average_coefficient;
    let learning_rate = next_latent_adam_learning_rate(step);
    update_adam_tensor_with_learning_rate(
        args.stream,
        args.optimizer,
        &mut args.weights.norm.weight,
        &args.grads.d_norm_weight,
        args.scratch,
        &mut args.state.norm.weight,
        step,
        average,
        learning_rate,
    )?;
    update_adam_tensor_with_learning_rate(
        args.stream,
        args.optimizer,
        &mut args.weights.norm.bias,
        &args.grads.d_norm_bias,
        args.scratch,
        &mut args.state.norm.bias,
        step,
        average,
        learning_rate,
    )?;
    update_linear_biases(args, step, average, learning_rate)
}

fn update_linear_biases(
    args: NextLatUpdateArgs<'_>,
    step: u32,
    average: f32,
    learning_rate: f32,
) -> Result<(), DriverError> {
    update_adam_tensor_with_learning_rate(
        args.stream,
        args.optimizer,
        &mut args.weights.input_projection.bias,
        &args.grads.d_input_projection_bias,
        args.scratch,
        &mut args.state.input_projection.bias,
        step,
        average,
        learning_rate,
    )?;
    update_adam_tensor_with_learning_rate(
        args.stream,
        args.optimizer,
        &mut args.weights.transition.bias,
        &args.grads.d_transition_bias,
        args.scratch,
        &mut args.state.transition.bias,
        step,
        average,
        learning_rate,
    )?;
    update_adam_tensor_with_learning_rate(
        args.stream,
        args.optimizer,
        &mut args.weights.output_projection.bias,
        &args.grads.d_output_projection_bias,
        args.scratch,
        &mut args.state.output_projection.bias,
        step,
        average,
        learning_rate,
    )
}
