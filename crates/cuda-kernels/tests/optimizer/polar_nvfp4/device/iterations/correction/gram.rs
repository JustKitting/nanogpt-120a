use std::error::Error;

use super::super::super::super::math::relative_l2;
use super::super::super::{CorrectionStats, Nvfp4Polar};
use super::super::{CorrectionGram, GramRequest};

impl<'a> Nvfp4Polar<'a> {
    pub(super) fn stale_correction_gram(
        &self,
        request: GramRequest<'_>,
        refresh: bool,
        defect_scale: f32,
        rejects_stale_steps: bool,
        stale_defect: &mut [f32],
        stats: &mut CorrectionStats,
    ) -> Result<CorrectionGram, Box<dyn Error>> {
        let GramRequest {
            source,
            rows,
            cols,
            iter,
        } = request;
        let gram_q = self.nvfp4_gram(source, rows, cols, iter, stats)?;
        let values = if refresh {
            let gram_hi = self.high_precision_gram(source, rows, cols, stats)?;
            for ((defect, q), hi) in stale_defect.iter_mut().zip(&gram_q).zip(&gram_hi) {
                *defect = q - hi;
            }
            stats.last_relative_defect = relative_l2(&gram_q, &gram_hi);
            stats.max_relative_defect = stats.max_relative_defect.max(stats.last_relative_defect);
            gram_hi
        } else {
            gram_q
                .into_iter()
                .zip(stale_defect.iter())
                .map(|(q, defect)| defect_scale.mul_add(-*defect, q))
                .collect()
        };

        Ok(CorrectionGram::new(
            values,
            rejects_stale_steps && !refresh,
            refresh,
        ))
    }

    pub(super) fn high_precision_gram(
        &self,
        source: &[f32],
        rows: usize,
        cols: usize,
        stats: &mut CorrectionStats,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        stats.high_precision_gram_count += 1;
        self.f16_product(source, source, rows, rows, cols)
    }

    pub(super) fn nvfp4_gram(
        &self,
        source: &[f32],
        rows: usize,
        cols: usize,
        iter: usize,
        stats: &mut CorrectionStats,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        stats.nvfp4_gram_count += 1;
        self.product(source, source, rows, rows, cols, iter, 0)
    }
}
