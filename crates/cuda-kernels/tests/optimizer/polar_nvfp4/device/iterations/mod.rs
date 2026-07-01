use std::error::Error;

use super::{CorrectionStats, GramCorrectionMode, Nvfp4Polar};
use correction::GramRequest;

mod correction;

fn row_orthogonality_residual(x: &[f32], rows: usize, cols: usize) -> f32 {
    let mut sum = 0.0_f32;
    for row in 0..rows {
        for col in 0..rows {
            let mut dot = 0.0_f32;
            for k in 0..cols {
                dot += x[row * cols + k] * x[col * cols + k];
            }
            let expected = if row == col { 1.0 } else { 0.0 };
            let diff = dot - expected;
            sum = diff.mul_add(diff, sum);
        }
    }
    sum.sqrt()
}

impl<'a> Nvfp4Polar<'a> {
    pub fn iterations(
        &self,
        mut source: Vec<f32>,
        rows: usize,
        cols: usize,
        iterations: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        for iter in 0..iterations {
            source = self.step(&source, rows, cols, iter)?;
        }
        Ok(source)
    }

    pub fn gram_corrected_iterations(
        &self,
        mut source: Vec<f32>,
        rows: usize,
        cols: usize,
        iterations: usize,
        mode: GramCorrectionMode,
    ) -> Result<(Vec<f32>, CorrectionStats), Box<dyn Error>> {
        let mut stats = CorrectionStats::pending();
        let mut stale_defect = vec![0.0_f32; rows * rows];

        for iter in 0..iterations {
            let request = GramRequest::new(&source, rows, cols, iter);
            let gram = self.correction_gram(request, mode, &mut stale_defect, &mut stats)?;
            source = self.step_from_gram(&source, &gram.values, rows, cols, iter)?;
        }

        Ok((source, stats))
    }

    pub fn gram_form_corrected_iterations(
        &self,
        mut source: Vec<f32>,
        rows: usize,
        cols: usize,
        iterations: usize,
        mode: GramCorrectionMode,
    ) -> Result<(Vec<f32>, CorrectionStats), Box<dyn Error>> {
        let mut stats = CorrectionStats::pending();
        let mut stale_defect = vec![0.0_f32; rows * rows];
        let mut residual = row_orthogonality_residual(&source, rows, cols);

        for iter in 0..iterations {
            let coefficient_safety = mode.coefficient_safety(iter);
            let request = GramRequest::new(&source, rows, cols, iter);
            let gram = self.correction_gram(request, mode, &mut stale_defect, &mut stats)?;

            let candidate = self.gram_form_step_from_gram(
                &source,
                &gram.values,
                rows,
                cols,
                iter,
                coefficient_safety,
            )?;
            let candidate_residual = row_orthogonality_residual(&candidate, rows, cols);
            if gram.stale_reject_candidate && candidate_residual > residual {
                stats.rejected_stale_steps += 1;
                continue;
            }

            source = candidate;
            residual = candidate_residual;
            if gram.refresh {
                residual = row_orthogonality_residual(&source, rows, cols);
            }
        }

        Ok((source, stats))
    }
}
