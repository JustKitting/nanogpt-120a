use cuda_core::DriverError;

use super::forward;
use super::tensors::{MlpForwardArgs, MlpProjectionTensors, MlpScratch};
use crate::random::InitRng;
use crate::types::{HiddenStateDevice, MlpDownLinear, MlpUpLinear};

#[derive(Clone, Debug)]
pub struct MlpWeights {
    pub c_fc: MlpUpLinear,
    pub c_proj: MlpDownLinear,
}

impl MlpWeights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        Self {
            c_fc: MlpUpLinear::init(rng),
            c_proj: MlpDownLinear::init(rng),
        }
    }

    pub fn input_from_attention<'a, 'scratch>(
        module: &'a rust_kernels_cuda::mlp::MlpModule,
        quant_module: &'a rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule,
        scratch: MlpScratch<'scratch>,
        projections: MlpProjectionTensors<'a>,
        hidden: HiddenStateDevice<'a>,
    ) -> MlpForwardArgs<'a, 'scratch> {
        MlpForwardArgs {
            module,
            quant_module,
            scratch,
            projections,
            hidden,
        }
    }

    pub fn forward<'a, 'scratch>(
        args: MlpForwardArgs<'a, 'scratch>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        forward::forward(args)
    }
}
