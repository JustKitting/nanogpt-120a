use std::sync::Arc;

use burn::train::metric::Adaptor;

use super::super::super::TrainStats;

#[derive(Clone)]
pub(in crate::training) struct CudaTrainOutput {
    pub(in crate::training) source: String,
    pub(in crate::training) window_offset: usize,
    pub(in crate::training) batch_size: usize,
    pub(in crate::training) seq_len: usize,
    pub(in crate::training) stats: Arc<TrainStats>,
}

impl burn::train::ItemLazy for CudaTrainOutput {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl Adaptor<CudaTrainOutput> for CudaTrainOutput {
    fn adapt(&self) -> CudaTrainOutput {
        self.clone()
    }
}

#[derive(Clone)]
pub(in crate::training) struct CudaValidOutput {
    pub(in crate::training::launch) val_loss: f32,
    pub(in crate::training::launch) eval_elapsed_s: f64,
    pub(in crate::training::launch) window_count: usize,
    pub(in crate::training::launch) completed_steps: usize,
}

impl burn::train::ItemLazy for CudaValidOutput {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl Adaptor<CudaValidOutput> for CudaValidOutput {
    fn adapt(&self) -> CudaValidOutput {
        self.clone()
    }
}
