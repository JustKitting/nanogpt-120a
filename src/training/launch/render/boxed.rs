use std::collections::HashSet;

use burn::train::metric::{MetricDefinition, MetricId};
use burn::train::renderer::{
    EvaluationName, EvaluationProgress, MetricState, MetricsRenderer, ProgressType,
    TrainingProgress,
};

pub(in crate::training::launch) struct BoxedMetricsRenderer {
    inner: Box<dyn MetricsRenderer>,
    hidden_metric_ids: HashSet<MetricId>,
}

impl BoxedMetricsRenderer {
    pub(in crate::training::launch) fn new(inner: Box<dyn MetricsRenderer>) -> Self {
        Self {
            inner,
            hidden_metric_ids: HashSet::new(),
        }
    }

    fn is_hidden(&self, state: &MetricState) -> bool {
        let metric_id = match state {
            MetricState::Generic(entry) => &entry.metric_id,
            MetricState::Numeric(entry, _) => &entry.metric_id,
        };
        self.hidden_metric_ids.contains(metric_id)
    }
}

impl burn::train::renderer::MetricsRendererTraining for BoxedMetricsRenderer {
    fn update_train(&mut self, state: MetricState) {
        if !self.is_hidden(&state) {
            self.inner.update_train(state);
        }
    }

    fn update_valid(&mut self, state: MetricState) {
        if !self.is_hidden(&state) {
            self.inner.update_valid(state);
        }
    }

    fn render_train(&mut self, item: TrainingProgress, progress_indicators: Vec<ProgressType>) {
        self.inner.render_train(item, progress_indicators);
    }

    fn render_valid(&mut self, item: TrainingProgress, progress_indicators: Vec<ProgressType>) {
        self.inner.render_valid(item, progress_indicators);
    }

    fn on_train_end(
        &mut self,
        summary: Option<burn::train::LearnerSummary>,
    ) -> Result<(), Box<dyn core::error::Error>> {
        self.inner.on_train_end(summary)
    }
}

impl burn::train::renderer::MetricsRendererEvaluation for BoxedMetricsRenderer {
    fn update_test(&mut self, name: EvaluationName, state: MetricState) {
        if !self.is_hidden(&state) {
            self.inner.update_test(name, state);
        }
    }

    fn render_test(&mut self, item: EvaluationProgress, progress_indicators: Vec<ProgressType>) {
        self.inner.render_test(item, progress_indicators);
    }

    fn on_test_end(
        &mut self,
        summary: Option<burn::train::LearnerSummary>,
    ) -> Result<(), Box<dyn core::error::Error>> {
        self.inner.on_test_end(summary)
    }
}

impl MetricsRenderer for BoxedMetricsRenderer {
    fn manual_close(&mut self) {
        self.inner.manual_close();
    }

    fn register_metric(&mut self, definition: MetricDefinition) {
        if hidden_renderer_metric(&definition.name) {
            self.hidden_metric_ids.insert(definition.metric_id);
        } else {
            self.inner.register_metric(definition);
        }
    }
}

fn hidden_renderer_metric(name: &str) -> bool {
    name.starts_with("Diagnostic ")
}
