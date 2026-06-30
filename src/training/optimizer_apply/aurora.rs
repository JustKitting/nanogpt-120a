use cuda_core::{CudaStream, DriverError};

use crate::training::runtime::Runtime;

use super::super::OptimizerTrace;
use super::super::optimizer_aurora::{AuroraMegaArgs, AuroraPointerTables, apply_aurora_mega};
use super::super::optimizer_tc_scratch::AuroraScratchBuffers;
use super::timed_ms;

pub(super) fn update_aurora_groups(
    stream: &CudaStream,
    runtime: &Runtime,
    tables: &AuroraPointerTables,
    scratch: &mut AuroraScratchBuffers,
    step: u32,
    average_coefficient: f32,
    trace: &mut OptimizerTrace,
) -> Result<(), DriverError> {
    trace.aurora_ms += timed_ms(|| {
        apply_aurora_mega(AuroraMegaArgs {
            stream,
            optimizer: &runtime.optimizer,
            table: &tables.all,
            scratch,
            slot_count: tables.slot_count,
            step,
            average_coefficient,
        })
    })?;
    Ok(())
}
