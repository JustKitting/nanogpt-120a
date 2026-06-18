use std::path::Path;

use super::Trainer;
use crate::AppResult;
use crate::checkpoint::save_uploaded_model;

impl Trainer {
    pub fn save_model(&self, path: &Path) -> AppResult {
        save_uploaded_model(self.runtime.stream.as_ref(), &self.uploaded, path)
    }
}
