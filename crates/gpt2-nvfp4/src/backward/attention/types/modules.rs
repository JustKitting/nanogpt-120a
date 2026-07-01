use rust_kernels_cuda::linear_backward::LinearBackwardModule;
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::transpose::TransposeModule;

#[derive(Clone, Copy)]
pub struct AttentionBackwardModules<'a> {
    pub transpose: &'a TransposeModule,
    pub decode: &'a Nvfp4DecodeModule,
    pub linear: &'a LinearBackwardModule,
    pub quant: &'a Nvfp4QuantModule,
}
