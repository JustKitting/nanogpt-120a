mod args;
mod config;
pub(crate) mod kernels;
mod launcher;

pub use args::{
    MsEdenDeviceScaleQuantArgs, MsEdenQuantArgs, Nvfp4QuantArgs, Nvfp4QuantRowwiseArgs,
    QuartetBackwardMsEdenDeviceScaleQuantArgs, QuartetBackwardMsEdenQuantArgs, RowAmaxArgs,
    TensorAmaxArgs,
};
pub use launcher::Nvfp4QuantModule;

pub const NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK: usize =
    kernels::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK as usize;

pub fn nvfp4_tensor_amax_chunks(element_count: usize) -> usize {
    element_count.div_ceil(NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK)
}
