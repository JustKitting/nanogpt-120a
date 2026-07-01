use rust_kernels_cuda::quartet;

pub use crate::common::nvfp4_tc::Nvfp4TcScratchBuffers as Scratch;

pub fn global_scale(x: &[f32]) -> f32 {
    let amax = x.iter().fold(0.0_f32, |acc, value| acc.max(value.abs()));
    quartet::quartet_backward_ms_eden_global_scale(amax)
}
