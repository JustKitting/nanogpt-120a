use std::sync::Arc;

use burn::train::metric::{Metric, MetricMetadata, MetricName, SerializedEntry};

pub(in crate::training) trait TextMetricSpec:
    Clone + Copy + Send + Sync
{
    type Input;

    fn name(self) -> &'static str;
    fn value(self, item: &Self::Input) -> String;
}

#[derive(Clone)]
pub(in crate::training) struct CudaTextMetric<S> {
    spec: S,
}

impl<S: TextMetricSpec> CudaTextMetric<S> {
    pub(in crate::training) fn new(spec: S) -> Self {
        Self { spec }
    }
}

impl<S: TextMetricSpec> Metric for CudaTextMetric<S> {
    type Input = S::Input;

    fn name(&self) -> MetricName {
        Arc::new(self.spec.name().to_string())
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let value = self.spec.value(item);
        SerializedEntry {
            formatted: value.clone(),
            serialized: value,
        }
    }

    fn clear(&mut self) {}
}
