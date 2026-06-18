use std::time::{Duration, Instant};

pub struct WallClockBudget {
    start: Instant,
    max: Option<Duration>,
}

impl WallClockBudget {
    pub fn new(max_seconds: Option<f64>) -> Self {
        Self {
            start: Instant::now(),
            max: max_seconds.map(Duration::from_secs_f64),
        }
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    pub fn expired(&self) -> bool {
        self.max
            .is_some_and(|max| self.start.elapsed().as_secs_f64() >= max.as_secs_f64())
    }
}
