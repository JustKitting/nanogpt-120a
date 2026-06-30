use std::sync::Arc;

use burn::train::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};

use super::metric_accumulator::MetricAccumulator;

pub(in crate::training) trait NumericMetricSpec:
    Clone + Copy + Send + Sync
{
    type Input;

    fn name(self) -> &'static str;
    fn unit(self) -> Option<&'static str>;
    fn higher_is_better(self) -> bool;
    fn value(self, item: &Self::Input) -> f64;
}

#[derive(Clone)]
pub(in crate::training) struct CudaNumericMetric<S> {
    spec: S,
    state: MetricAccumulator,
}

impl<S: NumericMetricSpec> CudaNumericMetric<S> {
    pub(in crate::training) fn new(spec: S) -> Self {
        Self {
            spec,
            state: MetricAccumulator::default(),
        }
    }
}

impl<S: NumericMetricSpec> Metric for CudaNumericMetric<S> {
    type Input = S::Input;

    fn name(&self) -> MetricName {
        Arc::new(self.spec.name().to_string())
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: self.spec.unit().map(str::to_string),
            higher_is_better: self.spec.higher_is_better(),
        }
        .into()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.state.update(self.spec.value(item), self.spec.unit())
    }

    fn clear(&mut self) {
        self.state.clear();
    }
}

impl<S: NumericMetricSpec> Numeric for CudaNumericMetric<S> {
    fn value(&self) -> NumericEntry {
        self.state.value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}
