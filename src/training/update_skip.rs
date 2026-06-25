use std::collections::VecDeque;

const ENABLED_ENV: &str = "TRAIN_SKIP_UNSTABLE_UPDATES";
const ROLLING_INTERVAL_ENV: &str = "TRAIN_SKIP_ROLLING_INTERVAL";
const SIGMA_FACTOR_ENV: &str = "TRAIN_SKIP_SIGMA_FACTOR";
const USE_LOSS_ENV: &str = "TRAIN_SKIP_USE_LOSS";
const USE_GRAD_NORM_ENV: &str = "TRAIN_SKIP_USE_GRAD_NORM";

const DEFAULT_ENABLED: bool = true;
const DEFAULT_ROLLING_INTERVAL: usize = 128;
const DEFAULT_SIGMA_FACTOR: f32 = 6.0;

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

#[derive(Clone, Copy)]
struct UpdateSkipConfig {
    enabled: bool,
    rolling_interval: usize,
    sigma_factor: f32,
    use_loss: bool,
    use_grad_norm: bool,
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

impl UpdateSkipConfig {
    fn from_env() -> Self {
        Self {
            enabled: env_bool(ENABLED_ENV, DEFAULT_ENABLED),
            rolling_interval: env_usize(ROLLING_INTERVAL_ENV, DEFAULT_ROLLING_INTERVAL).max(2),
            sigma_factor: env_f32(SIGMA_FACTOR_ENV, DEFAULT_SIGMA_FACTOR).max(0.0),
            use_loss: env_bool(USE_LOSS_ENV, true),
            use_grad_norm: env_bool(USE_GRAD_NORM_ENV, true),
        }
    }
}

fn push_history(history: &mut VecDeque<f32>, value: f32, max_len: usize) {
    history.push_back(value);
    while history.len() > max_len {
        history.pop_front();
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("1" | "true" | "yes" | "on") => true,
        Some("0" | "false" | "no" | "off") => false,
        _ => default,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &f32| value.is_finite())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> UpdateSkipState {
        UpdateSkipState {
            config: UpdateSkipConfig {
                enabled: true,
                rolling_interval: 4,
                sigma_factor: 2.0,
                use_loss: true,
                use_grad_norm: true,
            },
            losses: VecDeque::new(),
            grad_norms: VecDeque::new(),
        }
    }

    #[test]
    fn waits_for_minimum_history() {
        let mut state = state();
        assert!(!state.observe(Some(1.0), 1.0).skipped);
        assert!(!state.observe(Some(100.0), 100.0).skipped);
    }

    #[test]
    fn skips_loss_outlier_after_history() {
        let mut state = state();
        assert!(!state.observe(Some(1.0), 1.0).skipped);
        assert!(!state.observe(Some(1.1), 1.0).skipped);
        let decision = state.observe(Some(10.0), 1.0);
        assert!(decision.skipped);
        assert!(decision.loss_spike);
        assert!(!decision.grad_norm_spike);
    }

    #[test]
    fn skips_grad_norm_outlier_after_history() {
        let mut state = state();
        assert!(!state.observe(None, 1.0).skipped);
        assert!(!state.observe(None, 1.1).skipped);
        let decision = state.observe(None, 10.0);
        assert!(decision.skipped);
        assert!(!decision.loss_spike);
        assert!(decision.grad_norm_spike);
    }

    #[test]
    fn skips_non_finite_without_history() {
        let mut state = state();
        let decision = state.observe(Some(f32::NAN), 1.0);
        assert!(decision.skipped);
        assert!(decision.non_finite);
    }
}
