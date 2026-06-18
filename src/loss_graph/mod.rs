mod png;

use std::path::{Path, PathBuf};

use crate::AppResult;

pub(crate) struct LossCurve {
    points: Vec<LossPoint>,
}

pub(crate) struct LossPoint {
    pub step: usize,
    pub loss: f32,
    pub ema: f32,
}

impl LossCurve {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    pub fn push(&mut self, step: usize, loss: f32, ema: f32) {
        self.points.push(LossPoint { step, loss, ema });
    }

    pub fn write_png(&self, path: &Path) -> AppResult<PathBuf> {
        let path = png_path(path);
        if self.points.is_empty() {
            return Ok(path);
        }
        png::render(&path, &self.points)?;
        Ok(path)
    }
}

fn png_path(path: &Path) -> PathBuf {
    let mut path = path.to_path_buf();
    path.set_extension("png");
    path
}
