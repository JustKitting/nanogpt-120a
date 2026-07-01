use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

#[path = "linear_backward/args.rs"]
mod args;
#[path = "linear_backward/bias.rs"]
mod bias;
#[path = "linear_backward/device_scale.rs"]
mod device_scale;
#[path = "linear_backward/kernels.rs"]
mod kernels;
#[path = "linear_backward/ms_eden.rs"]
mod ms_eden;
pub use args::{
    LinearBackwardArgs, LinearBackwardDeviceScaleArgs, LinearBackwardInputTranspose,
    LinearBackwardMsEdenArgs, LinearBackwardMsEdenScratch, LinearBackwardMsEdenScratchBuffers,
    LinearBackwardWeightTranspose, MsEdenOperandScratch, MsEdenOperandScratchBuffer,
};
pub use bias::LINEAR_BIAS_THREADS_PER_BLOCK;

pub struct LinearBackwardModule {
    module: kernels::LoadedModule,
}

impl LinearBackwardModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn backward(&self, args: LinearBackwardArgs<'_, '_>) -> Result<(), DriverError> {
        let LinearBackwardArgs {
            stream,
            e_h,
            weight_t_h,
            e_t_h,
            input_t_h,
            dinput,
            dweight,
            dbias: _,
            token_count,
            input_dim,
            output_dim,
        } = args;

        self.backward_device_scale(LinearBackwardDeviceScaleArgs {
            stream,
            e_h,
            weight_t_h: weight_t_h.into(),
            e_t_h,
            input_t_h: input_t_h.into(),
            dinput,
            dweight,
            token_count,
            input_dim,
            output_dim,
        })
    }
}
