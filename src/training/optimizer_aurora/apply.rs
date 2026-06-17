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
            aurora_update: &args.scratch.oriented,
            fp32_workspace: &mut args.optimizer_scratch.fp32_workspace,
            amax: &mut args.optimizer_scratch.amax,
            next_global_scale: &mut args.optimizer_scratch.next_global_scale,
            len,
            learning_rate: AURORA_LR,
            weight_decay: AURORA_WEIGHT_DECAY,
        })?;
    args.tensor.global_scale = args
        .optimizer_scratch
        .next_global_scale
        .to_host_vec(args.stream)?[0];
    Ok(())
}
