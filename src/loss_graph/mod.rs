mod svg;

use std::fs;
use std::path::Path;

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

    pub fn write_svg(&self, path: &Path) -> AppResult {
        if self.points.is_empty() {
            return Ok(());
        }
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, svg::render(&self.points))?;
        Ok(())
    }
}
