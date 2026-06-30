use std::sync::{Arc, Mutex};

use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::BackendTypes;

use super::super::TokenDataLoader;
use super::{BurnBackend, BurnInnerBackend, CudaTrainInput, CudaValidInput};

#[derive(Clone)]
pub(super) struct CudaTrainDataLoader {
    data: Arc<Mutex<TokenDataLoader>>,
    total_steps: usize,
}

impl CudaTrainDataLoader {
    pub(super) fn new(data: TokenDataLoader, total_steps: usize) -> Self {
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

#[derive(Clone)]
pub(in crate::training) struct CudaValidationInput {
    pub(super) tokens: Arc<Vec<u16>>,
    pub(super) window_count: usize,
}

#[derive(Clone)]
pub(super) struct CudaValidDataLoader {
    input: CudaValidationInput,
}

impl CudaValidDataLoader {
    pub(super) fn new(tokens: Vec<u16>, window_count: usize) -> Self {
        Self {
            input: CudaValidationInput {
                tokens: Arc::new(tokens),
                window_count,
            },
        }
    }
}

impl DataLoader<BurnInnerBackend, CudaValidInput> for CudaValidDataLoader {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<CudaValidInput> + 'a> {
        Box::new(CudaValidIterator {
            input: self.input.clone(),
            produced: false,
        })
    }

    fn num_items(&self) -> usize {
        self.input.window_count
    }

    fn to_device(
        &self,
        _device: &<BurnInnerBackend as BackendTypes>::Device,
    ) -> Arc<dyn DataLoader<BurnInnerBackend, CudaValidInput>> {
        Arc::new(self.clone())
    }

    fn slice(
        &self,
        _start: usize,
        _end: usize,
    ) -> Arc<dyn DataLoader<BurnInnerBackend, CudaValidInput>> {
        Arc::new(self.clone())
    }
}

struct CudaValidIterator {
    input: CudaValidationInput,
    produced: bool,
}

impl Iterator for CudaValidIterator {
    type Item = CudaValidInput;

    fn next(&mut self) -> Option<Self::Item> {
        if self.produced {
            return None;
        }
        self.produced = true;
        Some(Ok(self.input.clone()))
    }
}

impl DataLoaderIterator<CudaValidInput> for CudaValidIterator {
    fn progress(&self) -> Progress {
        Progress::new(usize::from(self.produced), 1)
    }
}
