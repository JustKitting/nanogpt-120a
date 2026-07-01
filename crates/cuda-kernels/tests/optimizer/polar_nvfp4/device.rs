use cuda_core::CudaStream;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tc_matmul::Nvfp4TcMatmulModule;

#[path = "device/iterations/mod.rs"]
mod iterations;
#[path = "device/mode.rs"]
mod mode;
#[path = "device/product.rs"]
mod product;
#[path = "device/stats.rs"]
mod stats;
#[path = "device/step.rs"]
mod step;

pub use mode::GramCorrectionMode;
pub use stats::CorrectionStats;

pub struct Nvfp4Polar<'a> {
    stream: &'a CudaStream,
    f16: &'a F16TcMatmulModule,
    matmul: &'a Nvfp4TcMatmulModule,
    quant: &'a Nvfp4QuantModule,
}

impl<'a> Nvfp4Polar<'a> {
    pub fn new(
        stream: &'a CudaStream,
        f16: &'a F16TcMatmulModule,
        matmul: &'a Nvfp4TcMatmulModule,
        quant: &'a Nvfp4QuantModule,
    ) -> Self {
        Self {
            stream,
            f16,
            matmul,
            quant,
        }
    }
}
