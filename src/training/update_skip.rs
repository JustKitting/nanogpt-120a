use std::collections::VecDeque;

mod config;
#[cfg(test)]
mod tests;

use config::UpdateSkipConfig;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct UpdateSkipDecision {
    pub skipped: bool,
    pub loss_spike: bool,
    pub grad_norm_spike: bool,
    pub non_finite: bool,
}

pub(super) struct UpdateSkipState {
    config: UpdateSkipConfig,
    losses: VecDeque<f32>,
    grad_norms: VecDeque<f32>,
}

impl UpdateSkipState {
    pub(super) fn new() -> Self {
        Self {
            config: UpdateSkipConfig::from_env(),
            losses: VecDeque::new(),
            grad_norms: VecDeque::new(),
        }
    }

    pub(super) fn observe(&mut self, loss: Option<f32>, grad_norm: f32) -> UpdateSkipDecision {
        if !self.config.enabled {
            return UpdateSkipDecision::default();
        }

        let loss_non_finite = loss.is_some_and(|value| !value.is_finite());
        let grad_non_finite = !grad_norm.is_finite();
        let loss_spike = self.config.use_loss
            && loss
                .filter(|value| value.is_finite())
                .is_some_and(|value| self.is_spike(value, &self.losses));
        let grad_norm_spike = self.config.use_grad_norm
            && grad_norm.is_finite()
            && self.is_spike(grad_norm, &self.grad_norms);

        if let Some(loss) = loss.filter(|value| value.is_finite()) {
            push_history(&mut self.losses, loss, self.config.rolling_interval);
        }
        if grad_norm.is_finite() {
            push_history(
                &mut self.grad_norms,
                grad_norm,
                self.config.rolling_interval,
            );
        }

        let non_finite = loss_non_finite || grad_non_finite;
        UpdateSkipDecision {
            skipped: non_finite || loss_spike || grad_norm_spike,
            loss_spike,
            grad_norm_spike,
            non_finite,
        }
    }

    fn is_spike(&self, value: f32, history: &VecDeque<f32>) -> bool {
        if history.len() < self.min_history() {
            return false;
        }

        let len = history.len() as f32;
        let mean = history.iter().sum::<f32>() / len;
        let variance = history
            .iter()
            .map(|sample| {
                let diff = *sample - mean;
                diff * diff
            })
            .sum::<f32>()
            / len;
        let threshold = mean + self.config.sigma_factor * variance.sqrt();
        value > threshold
    }

    fn min_history(&self) -> usize {
        (self.config.rolling_interval / 2).max(2)
    }
}

fn push_history(history: &mut VecDeque<f32>, value: f32, max_len: usize) {
    history.push_back(value);
    while history.len() > max_len {
        history.pop_front();
    }
}
