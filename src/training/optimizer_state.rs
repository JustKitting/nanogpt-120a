use super::update_skip::{UpdateSkipDecision, UpdateSkipState};

mod device;
mod init;
mod types;

pub use types::OptimizerStateBuffers;
pub(in crate::training) use types::{
    AdamState, AuroraState, BlockState, LayerNormState, LinearState, NextLatState,
};

impl OptimizerStateBuffers {
    pub(super) fn advance(&mut self) -> u32 {
        self.step += 1;
        self.step
    }

    pub(super) fn schedule_free_average_coefficient(&mut self, step: u32) -> f32 {
        super::learning_rate::schedule_free_average_coefficient(
            step,
            &mut self.schedule_free_weight_sum,
        )
    }

    pub(super) fn next_step(&self) -> u32 {
        self.step + 1
    }

    pub(super) fn should_skip_update(
        &mut self,
        loss: Option<f32>,
        grad_norm: f32,
    ) -> UpdateSkipDecision {
        self.update_skip.observe(loss, grad_norm)
    }
}
