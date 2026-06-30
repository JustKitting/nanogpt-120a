use std::time::{Duration, Instant};

pub(super) struct WallClockBudget {
    start: Instant,
    max: Duration,
}

impl WallClockBudget {
    pub(super) fn new(max_seconds: f64) -> Self {
        Self {
            start: Instant::now(),
            max: Duration::from_secs_f64(max_seconds),
        }
    }

    pub(super) fn elapsed_seconds(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    pub(super) fn expired(&self) -> bool {
        self.start.elapsed() >= self.max
    }
}
