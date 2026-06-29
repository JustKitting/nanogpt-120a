#[derive(Clone, Debug)]
pub struct SweepRng {
    state: u64,
}

impl SweepRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (u64::MAX as f64);
        self.next_u64() as f64 * SCALE
    }

    pub fn usize(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }

    pub fn choose<T: Copy>(&mut self, values: &[T]) -> T {
        values[self.usize(values.len())]
    }

    pub fn log_uniform(&mut self, min: f64, max: f64) -> f64 {
        let lo = min.ln();
        let hi = max.ln();
        (lo + (hi - lo) * self.f64()).exp()
    }
}
