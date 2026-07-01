use std::error::Error;

use super::{CorrectionStats, Nvfp4Polar};
use crate::polar_nvfp4::math::{combine_next, transpose};

impl<'a> Nvfp4Polar<'a> {
    pub fn step(
        &self,
        source: &[f32],
        rows: usize,
        cols: usize,
        iter: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let gram = self.product(source, source, rows, rows, cols, iter, 0)?;
        let source_t = transpose(source, rows, cols);
        let ax = self.product(&gram, &source_t, rows, cols, rows, iter, 1)?;
        let ax_t = transpose(&ax, rows, cols);
        let aax = self.product(&gram, &ax_t, rows, cols, rows, iter, 2)?;
        Ok(combine_next(source, &ax, &aax, iter))
    }

    pub(super) fn step_from_gram(
        &self,
        source: &[f32],
        gram: &[f32],
        rows: usize,
        cols: usize,
        iter: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let ax = self.f16_product(gram, &transpose(source, rows, cols), rows, cols, rows)?;
        let aax = self.f16_product(gram, &transpose(&ax, rows, cols), rows, cols, rows)?;
        Ok(combine_next(source, &ax, &aax, iter))
    }

    pub(super) fn gram_form_step_from_gram(
        &self,
        source: &[f32],
        gram: &[f32],
        rows: usize,
        cols: usize,
        iter: usize,
        coefficient_safety: f32,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let (mut a, mut b, mut c) = crate::polar_coefficients::coefficients(iter);
        if coefficient_safety != 1.0 {
            a /= coefficient_safety;
            b /= coefficient_safety.powi(3);
            c /= coefficient_safety.powi(5);
        }
        let gram2 = self.f16_product(gram, &transpose(gram, rows, rows), rows, rows, rows)?;
        let mut q = vec![0.0_f32; rows * rows];
        for index in 0..q.len() {
            q[index] = c.mul_add(gram2[index], b * gram[index]);
        }
        for row in 0..rows {
            q[row * rows + row] += a;
        }
        self.f16_product(&q, &transpose(source, rows, cols), rows, cols, rows)
    }

    pub(super) fn averaged_nvfp4_gram(
        &self,
        source: &[f32],
        rows: usize,
        cols: usize,
        iter: usize,
        samples: usize,
        stats: &mut CorrectionStats,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut sum = vec![0.0_f32; rows * rows];
        for sample in 0..samples {
            stats.nvfp4_gram_count += 1;
            let gram = self.product(source, source, rows, rows, cols, iter, 16 + sample as u32)?;
            for (sum, value) in sum.iter_mut().zip(gram) {
                *sum += value;
            }
        }
        let inv_samples = 1.0 / samples as f32;
        for value in &mut sum {
            *value *= inv_samples;
        }
        Ok(sum)
    }
}
