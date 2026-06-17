use cuda_core::DriverError;
use rust_kernels_cuda::optimizer::Nvfp4WeightUpdateArgs;

use super::{AURORA_LR, AURORA_WEIGHT_DECAY, AuroraMatrixArgs};

pub(super) fn apply_update(args: AuroraMatrixArgs<'_, '_>, len: u32) -> Result<(), DriverError> {
    args.modules
        .optimizer
        .apply_nvfp4_weight_update(Nvfp4WeightUpdateArgs {
            stream: args.stream,
            bytes: &mut args.tensor.bytes,
            scales: &mut args.tensor.scales,
            global_scale: args.tensor.global_scale,
            requantize_global_scale: 0.0,
            aurora_update: &args.scratch.oriented,
            fp32_workspace: &mut args.optimizer_scratch.fp32_workspace,
            amax: &mut args.optimizer_scratch.amax,
            chunk_amax: &mut args.optimizer_scratch.chunk_amax,
            next_global_scale: &mut args.optimizer_scratch.next_global_scale,
            len,
            learning_rate: AURORA_LR * super::super::learning_rate::aurora_scale(),
            weight_decay: AURORA_WEIGHT_DECAY,
        })?;
    args.tensor.global_scale = args
        .optimizer_scratch
        .next_global_scale
        .to_host_vec(args.stream)?[0];
    Ok(())
}
