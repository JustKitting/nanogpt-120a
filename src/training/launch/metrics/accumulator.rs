use burn::train::metric::{NumericEntry, SerializedEntry};

#[derive(Clone, Default)]
pub(super) struct MetricAccumulator {
    current: f64,
    sum: f64,
    count: usize,
}

impl MetricAccumulator {
    pub(super) fn update(&mut self, value: f64, unit: Option<&str>) -> SerializedEntry {
        self.current = value;
        if value.is_finite() {
            self.sum += value;
            self.count += 1;
        }
        SerializedEntry {
            formatted: format_metric_value(value, unit),
            serialized: value.to_string(),
        }
    }

    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }

    pub(super) fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current)
    }

    pub(super) fn running_value(&self) -> NumericEntry {
        if self.count == 0 {
            NumericEntry::Value(f64::NAN)
        } else {
            NumericEntry::Aggregated {
                aggregated_value: self.sum / self.count as f64,
                count: self.count,
            }
        }
    }
}

fn format_metric_value(value: f64, unit: Option<&str>) -> String {
    match unit {
        Some(unit) => format!("{value:.6} {unit}"),
        None => format!("{value:.6}"),
    }
}
