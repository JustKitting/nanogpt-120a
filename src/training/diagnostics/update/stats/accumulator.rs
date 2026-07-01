#[derive(Default)]
pub(super) struct UpdateStatAccumulator {
    pub(super) grad_sum_sq: f64,
    pub(super) weight_before_sum_sq: f64,
    pub(super) weight_after_sum_sq: f64,
    pub(super) delta_sum_sq: f64,
    pub(super) grad_dot_delta: f64,
    pub(super) predicted_delta_sum_sq: f64,
    pub(super) grad_dot_predicted_delta: f64,
    pub(super) quant_error_sum_sq: f64,
    pub(super) grad_max: f32,
    pub(super) delta_max: f32,
    pub(super) grad_nonzero: usize,
    pub(super) grad_finite: bool,
}

impl UpdateStatAccumulator {
    pub(super) fn new() -> Self {
        Self {
            grad_finite: true,
            ..Self::default()
        }
    }

    pub(super) fn record(
        &mut self,
        grad: f32,
        before: f32,
        after: f32,
        delta: f32,
        predicted_delta: f32,
        quant_error: f32,
    ) {
        self.grad_finite &= grad.is_finite();
        if grad != 0.0 {
            self.grad_nonzero += 1;
        }
        self.grad_max = self.grad_max.max(grad.abs());
        self.delta_max = self.delta_max.max(delta.abs());
        self.grad_sum_sq += square(grad);
        self.weight_before_sum_sq += square(before);
        self.weight_after_sum_sq += square(after);
        self.delta_sum_sq += square(delta);
        self.grad_dot_delta += product(grad, delta);
        self.predicted_delta_sum_sq += square(predicted_delta);
        self.grad_dot_predicted_delta += product(grad, predicted_delta);
        self.quant_error_sum_sq += square(quant_error);
    }
}

pub(super) fn rms(sum_sq: f64, len: f64) -> f32 {
    (sum_sq / len).sqrt() as f32
}

pub(super) fn ratio_or_zero(numerator: f32, denominator: f32) -> f32 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

pub(super) fn cosine_or_zero(dot: f64, lhs_sum_sq: f64, rhs_sum_sq: f64) -> f32 {
    if lhs_sum_sq > 0.0 && rhs_sum_sq > 0.0 {
        (dot / (lhs_sum_sq.sqrt() * rhs_sum_sq.sqrt())) as f32
    } else {
        0.0
    }
}

fn square(value: f32) -> f64 {
    product(value, value)
}

fn product(lhs: f32, rhs: f32) -> f64 {
    (lhs as f64) * (rhs as f64)
}
