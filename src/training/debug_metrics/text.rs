use burn::train::SupervisedTraining;

use super::super::launch::{CudaLearningComponents, CudaTrainOutput};
use crate::training::text_metric::{CudaTextMetric, TextMetricSpec};

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

impl TextMetricSpec for DebugTextField {
    type Input = CudaTrainOutput;

    fn name(self) -> &'static str {
        match self {
            Self::TokenEmbeddingHashBefore => "Diagnostic token embedding hash before",
            Self::TokenEmbeddingHashAfter => "Diagnostic token embedding hash after",
            Self::TensorNames => "Diagnostic tensor names",
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

pub(super) fn register_text_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    for field in DEBUG_TEXT_FIELDS {
        training = training.metric_train(CudaTextMetric::new(*field));
    }
    training
}
