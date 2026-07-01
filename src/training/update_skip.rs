mod config;
mod rolling;
#[cfg(test)]
mod tests;

use config::UpdateSkipConfig;
use rolling::RollingHistory;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct UpdateSkipDecision {
    pub skipped: bool,
    pub loss_spike: bool,
    pub grad_norm_spike: bool,
    pub non_finite: bool,
}

pub(super) struct UpdateSkipState {
    config: UpdateSkipConfig,
    losses: RollingHistory,
    grad_norms: RollingHistory,
}

impl UpdateSkipState {
    pub(super) fn new() -> Self {
        Self::from_config(UpdateSkipConfig::from_env())
    }

    fn from_config(config: UpdateSkipConfig) -> Self {
        Self {
            losses: RollingHistory::new(config.rolling_interval),
            grad_norms: RollingHistory::new(config.rolling_interval),
            config,
        }
    }

    pub(super) fn observe(&mut self, loss: Option<f32>, grad_norm: f32) -> UpdateSkipDecision {
        if !self.config.enabled {
            return UpdateSkipDecision::default();
        }

        let finite_loss = loss.filter(|value| value.is_finite());
        let loss_non_finite = loss.is_some() && finite_loss.is_none();
        let grad_non_finite = !grad_norm.is_finite();
        let loss_spike = self.config.use_loss
            && finite_loss
                .is_some_and(|value| self.losses.is_spike(value, self.config.sigma_factor));
        let grad_norm_spike = self.config.use_grad_norm
            && grad_norm.is_finite()
            && self
                .grad_norms
                .is_spike(grad_norm, self.config.sigma_factor);

        if let Some(loss) = finite_loss {
            self.losses.push(loss);
        }
        if grad_norm.is_finite() {
            self.grad_norms.push(grad_norm);
        }

        let non_finite = loss_non_finite || grad_non_finite;
        UpdateSkipDecision {
            skipped: non_finite || loss_spike || grad_norm_spike,
            loss_spike,
            grad_norm_spike,
            non_finite,
        }
    }
}
