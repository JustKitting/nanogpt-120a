use std::collections::VecDeque;

pub(super) struct RollingHistory {
    values: VecDeque<f32>,
    max_len: usize,
    min_len: usize,
}

impl RollingHistory {
    pub(super) fn new(max_len: usize) -> Self {
        Self {
            values: VecDeque::new(),
            max_len,
            min_len: (max_len / 2).max(2),
        }
    }

    pub(super) fn push(&mut self, value: f32) {
        self.values.push_back(value);
        while self.values.len() > self.max_len {
            self.values.pop_front();
        }
    }

    pub(super) fn is_spike(&self, value: f32, sigma_factor: f32) -> bool {
        let Some((mean, variance)) = self.stats() else {
            return false;
        };
        value > mean + sigma_factor * variance.sqrt()
    }

    fn stats(&self) -> Option<(f32, f32)> {
        if self.values.len() < self.min_len {
            return None;
        }

        let len = self.values.len() as f32;
        let mean = self.values.iter().sum::<f32>() / len;
        let variance = self
            .values
            .iter()
            .map(|sample| {
                let diff = *sample - mean;
                diff * diff
            })
            .sum::<f32>()
            / len;
        Some((mean, variance))
    }
}
