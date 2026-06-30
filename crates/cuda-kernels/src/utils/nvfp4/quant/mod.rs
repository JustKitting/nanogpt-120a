mod amax;
mod args;
mod config;
mod four_six;
mod global_scale;
pub(crate) mod kernels;
mod launcher;
mod shape;

pub use args::{
    MsEdenDeviceScaleQuantArgs, MsEdenPairDeviceScaleQuantArgs, MsEdenQuantArgs,
    MsEdenTransposeDeviceScaleQuantArgs, Nvfp4QuantArgs, Nvfp4QuantRowwiseArgs,
    Nvfp4TransposeMsEdenDeviceScaleQuantArgs, QuartetBackwardMsEdenDeviceScaleQuantArgs,
    QuartetBackwardMsEdenQuantArgs, RowAmaxArgs, RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs,
    TensorAmaxArgs,
};
pub use launcher::Nvfp4QuantModule;

pub const NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK: usize =
    kernels::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK as usize;
