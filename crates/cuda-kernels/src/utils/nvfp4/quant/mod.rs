mod args;
mod config;
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
