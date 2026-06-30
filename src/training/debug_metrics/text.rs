use std::sync::Arc;

use burn::train::SupervisedTraining;
use burn::train::metric::{Metric, MetricMetadata, MetricName, SerializedEntry};

use super::super::launch::{CudaLearningComponents, CudaTrainOutput};

#[derive(Clone, Copy)]
struct DebugTextMetricSpec {
    name: &'static str,
    field: DebugTextField,
}

#[derive(Clone, Copy)]
enum DebugTextField {
    TokenEmbeddingHashBefore,
    TokenEmbeddingHashAfter,
    TensorNames,
}

const DEBUG_TEXT_FIELDS: &[DebugTextField] = &[
    DebugTextField::TokenEmbeddingHashBefore,
    DebugTextField::TokenEmbeddingHashAfter,
    DebugTextField::TensorNames,
];

impl DebugTextField {
    const fn spec(self) -> DebugTextMetricSpec {
        match self {
            Self::TokenEmbeddingHashBefore => DebugTextMetricSpec {
                name: "Diagnostic token embedding hash before",
                field: self,
            },
            Self::TokenEmbeddingHashAfter => DebugTextMetricSpec {
                name: "Diagnostic token embedding hash after",
                field: self,
            },
            Self::TensorNames => DebugTextMetricSpec {
                name: "Diagnostic tensor names",
                field: self,
            },
        }
    }

    fn value(self, item: &CudaTrainOutput) -> String {
        let Some(trace) = item.stats.diagnostics.as_ref() else {
            return String::new();
        };

        match self {
            Self::TokenEmbeddingHashBefore => {
                format!("{:016x}", trace.token_embedding_hash_before)
            }
            Self::TokenEmbeddingHashAfter => {
                format!("{:016x}", trace.token_embedding_hash_after)
            }
            Self::TensorNames => trace
                .updates
                .iter()
                .map(|update| update.name.as_str())
                .collect::<Vec<_>>()
                .join(","),
        }
    }
}

#[derive(Clone)]
struct CudaDebugTextMetric {
    spec: DebugTextMetricSpec,
}

impl CudaDebugTextMetric {
    fn new(spec: DebugTextMetricSpec) -> Self {
        Self { spec }
    }
}

impl Metric for CudaDebugTextMetric {
    type Input = CudaTrainOutput;

    fn name(&self) -> MetricName {
        Arc::new(self.spec.name.to_string())
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let value = self.spec.field.value(item);
        SerializedEntry {
            formatted: value.clone(),
            serialized: value,
        }
    }

    fn clear(&mut self) {}
}

pub(super) fn register_text_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    for field in DEBUG_TEXT_FIELDS {
        training = training.metric_train(CudaDebugTextMetric::new(field.spec()));
    }
    training
}
