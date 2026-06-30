use std::sync::{Arc, Mutex};

use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::BackendTypes;

use super::super::{BurnBackend, CudaTrainInput};
use crate::training::TokenDataLoader;

#[derive(Clone)]
pub(in crate::training::launch) struct CudaTrainDataLoader {
    data: Arc<Mutex<TokenDataLoader>>,
    total_steps: usize,
}

impl CudaTrainDataLoader {
    pub(in crate::training::launch) fn new(data: TokenDataLoader, total_steps: usize) -> Self {
        Self {
            data: Arc::new(Mutex::new(data)),
            total_steps,
        }
    }
}

impl DataLoader<BurnBackend, CudaTrainInput> for CudaTrainDataLoader {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<CudaTrainInput> + 'a> {
        Box::new(CudaTrainIterator {
            data: Arc::clone(&self.data),
            produced: 0,
            total_steps: self.total_steps,
        })
    }

    fn num_items(&self) -> usize {
        self.total_steps
    }

    fn to_device(
        &self,
        _device: &<BurnBackend as BackendTypes>::Device,
    ) -> Arc<dyn DataLoader<BurnBackend, CudaTrainInput>> {
        Arc::new(self.clone())
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<BurnBackend, CudaTrainInput>> {
        let mut sliced = self.clone();
        sliced.total_steps = end.saturating_sub(start).min(self.total_steps);
        Arc::new(sliced)
    }
}

struct CudaTrainIterator {
    data: Arc<Mutex<TokenDataLoader>>,
    produced: usize,
    total_steps: usize,
}

impl Iterator for CudaTrainIterator {
    type Item = CudaTrainInput;

    fn next(&mut self) -> Option<Self::Item> {
        if self.produced >= self.total_steps {
            return None;
        }
        self.produced += 1;
        Some(
            self.data
                .lock()
                .map_err(|err| err.to_string())
                .and_then(|mut data| data.next_batch().map_err(|err| err.to_string())),
        )
    }
}

impl DataLoaderIterator<CudaTrainInput> for CudaTrainIterator {
    fn progress(&self) -> Progress {
        Progress::new(self.produced, self.total_steps)
    }
}
