use std::error::Error;

mod common;
#[path = "ms_eden_transpose/fp32.rs"]
mod fp32;
#[path = "ms_eden_transpose/nvfp4.rs"]
mod nvfp4;
#[path = "ms_eden_transpose/rowwise.rs"]
mod rowwise;
#[path = "ms_eden_transpose/support/mod.rs"]
mod support;

type TestResult = Result<(), Box<dyn Error>>;
