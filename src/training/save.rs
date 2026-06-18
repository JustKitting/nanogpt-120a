use std::path::Path;

use super::Trainer;
use super::optimizer_state::OptimizerStateBuffers;
use crate::AppResult;
use crate::checkpoint::{load_uploaded_model, save_uploaded_model};

impl Trainer {
    pub fn save_model(&self, path: &Path) -> AppResult {
        save_uploaded_model(self.runtime.stream.as_ref(), &self.uploaded, path)
    }

    pub fn load_model(&mut self, path: &Path) -> AppResult {
        let stream = self.runtime.stream.as_ref();
        let uploaded = load_uploaded_model(stream, path)?;
        self.buffers.optimizer_state =
            OptimizerStateBuffers::new(stream, &self.runtime.decode, &uploaded)?;
        self.uploaded = uploaded;
        Ok(())
    }
}
