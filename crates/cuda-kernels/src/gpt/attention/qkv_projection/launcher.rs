use cuda_core::DriverError;

use super::super::AttentionModule;
use super::args::{CProjArgs, QkvProjectionArgs};
use super::config::{c_proj_params, cta_config, qkv_params};

impl AttentionModule {
    pub fn qkv_projection(&self, args: QkvProjectionArgs<'_, '_>) -> Result<(), DriverError> {
        self.qkv_projection.attention_projection_kernel(
            args.stream,
            cta_config(args.token_count, args.output_dim),
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.weight.bytes,
            args.weight.scales,
            args.bias.bytes,
            args.bias.scales,
            args.weight.global_scale,
            args.bias.global_scale,
            args.out,
            qkv_params(&args),
        )
    }

    pub fn c_proj(&self, args: CProjArgs<'_, '_>) -> Result<(), DriverError> {
        self.qkv_projection.attention_projection_kernel(
            args.stream,
            cta_config(args.token_count, args.embedding_dim),
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.weight.bytes,
            args.weight.scales,
            args.bias.bytes,
            args.bias.scales,
            args.weight.global_scale,
            args.bias.global_scale,
            args.residual,
            c_proj_params(args.token_count, args.embedding_dim),
        )
    }
}
