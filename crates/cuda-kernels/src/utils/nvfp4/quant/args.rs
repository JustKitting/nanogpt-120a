#[path = "args/basic.rs"]
mod basic;
#[path = "args/ms_eden.rs"]
mod ms_eden;
#[path = "args/ms_eden_transpose.rs"]
mod ms_eden_transpose;

pub use basic::{Nvfp4QuantArgs, Nvfp4QuantRowwiseArgs, RowAmaxArgs, TensorAmaxArgs};
pub use ms_eden::{
    MsEdenDeviceScaleQuantArgs, MsEdenPairDeviceScaleQuantArgs, MsEdenQuantArgs,
    MsEdenTransposeDeviceScaleQuantArgs, QuartetBackwardMsEdenDeviceScaleQuantArgs,
    QuartetBackwardMsEdenQuantArgs,
};
pub use ms_eden_transpose::{
    Nvfp4TransposeMsEdenDeviceScaleQuantArgs, RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs,
};
