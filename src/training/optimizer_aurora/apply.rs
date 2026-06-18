use cuda_core::DriverError;
use rust_kernels_cuda::optimizer::Nvfp4WeightUpdateArgs;

use super::{AURORA_WEIGHT_DECAY, AuroraMatrixArgs, aurora_learning_rate};

pub(super) fn apply_update(args: AuroraMatrixArgs<'_, '_>, len: u32) -> Result<(), DriverError> {
    args.modules
        .optimizer
        .apply_nvfp4_weight_update(Nvfp4WeightUpdateArgs {
            stream: args.stream,
            bytes: &mut args.tensor.bytes,
            scales: &mut args.tensor.scales,
            global_scale: &mut args.tensor.global_scale,
            z_master: &mut args.state.z_master,
            x_master: &mut args.state.x_master,
            aurora_update: &args.scratch.oriented,
            amax: &mut args.optimizer_scratch.amax,
            chunk_amax: &mut args.optimizer_scratch.chunk_amax,
            len,
            learning_rate: aurora_learning_rate(args.step),
            weight_decay: AURORA_WEIGHT_DECAY,
            average_coefficient: args.average_coefficient,
        })?;
    Ok(())
}
