use cuda_core::CudaStream;

use crate::linear_backward::LinearBackwardMsEdenArgs;
use crate::nvfp4_quant::Nvfp4QuantModule;
use crate::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;

pub(in crate::linear_backward::ms_eden) struct QuantizeContext<'a> {
    pub(in crate::linear_backward::ms_eden::quantize) module: &'a Nvfp4QuantModule,
    pub(in crate::linear_backward::ms_eden::quantize) stream: &'a CudaStream,
    pub(in crate::linear_backward::ms_eden::quantize) token_count: u32,
    pub(in crate::linear_backward::ms_eden::quantize) input_dim: u32,
    pub(in crate::linear_backward::ms_eden::quantize) output_dim: u32,
    pub(in crate::linear_backward::ms_eden::quantize) output_k: u32,
    pub(in crate::linear_backward::ms_eden::quantize) token_k: u32,
    pub(in crate::linear_backward::ms_eden::quantize) sign_seed: u32,
    pub(in crate::linear_backward::ms_eden::quantize) scale_seed: u32,
}

impl<'a> QuantizeContext<'a> {
    pub(in crate::linear_backward::ms_eden) fn for_args<'scratch, 'out>(
        args: &LinearBackwardMsEdenArgs<'a, 'scratch, 'out>,
    ) -> Self {
        Self {
            module: args.quant_module,
            stream: args.stream,
            token_count: args.token_count,
            input_dim: args.input_dim,
            output_dim: args.output_dim,
            output_k: nvfp4_tc_matmul_padded_k(args.output_dim),
            token_k: nvfp4_tc_matmul_padded_k(args.token_count),
            sign_seed: args.sign_seed,
            scale_seed: args.scale_seed,
        }
    }
}
