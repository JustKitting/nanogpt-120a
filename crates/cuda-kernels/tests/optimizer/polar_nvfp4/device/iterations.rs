use std::error::Error;

use super::{CorrectionStats, GramCorrectionMode, Nvfp4Polar, row_orthogonality_residual};

#[derive(Clone, Copy)]
struct GramRequest<'s> {
    source: &'s [f32],
    rows: usize,
    cols: usize,
    iter: usize,
}

struct CorrectionGram {
    values: Vec<f32>,
    stale_reject_candidate: bool,
    refresh: bool,
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
        let mut stats = CorrectionStats {
            last_relative_defect: f32::INFINITY,
            ..CorrectionStats::default()
        };
        let mut stale_defect = vec![0.0_f32; rows * rows];

        for iter in 0..iterations {
            let request = GramRequest {
                source: &source,
                rows,
                cols,
                iter,
            };
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
        let mut stats = CorrectionStats {
            last_relative_defect: f32::INFINITY,
            ..CorrectionStats::default()
        };
        let mut stale_defect = vec![0.0_f32; rows * rows];
        let mut residual = row_orthogonality_residual(&source, rows, cols);

        for iter in 0..iterations {
            let coefficient_safety = mode.coefficient_safety(iter);
            let request = GramRequest {
                source: &source,
                rows,
                cols,
                iter,
            };
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

    fn correction_gram(
        &self,
        request: GramRequest<'_>,
        mode: GramCorrectionMode,
        stale_defect: &mut [f32],
        stats: &mut CorrectionStats,
    ) -> Result<CorrectionGram, Box<dyn Error>> {
        let GramRequest {
            source,
            rows,
            cols,
            iter,
        } = request;
        let rejects_stale_steps = mode.rejects_stale_steps();
        let gram = match mode {
            GramCorrectionMode::HighPrecision | GramCorrectionMode::HighPrecisionSafety { .. } => {
                CorrectionGram {
                    values: self.high_precision_gram(source, rows, cols, stats)?,
                    stale_reject_candidate: false,
                    refresh: true,
                }
            }
            GramCorrectionMode::Nvfp4GramOnly
            | GramCorrectionMode::Nvfp4GramOnlySafety { .. }
            | GramCorrectionMode::Nvfp4GramOnlySchedule { .. }
            | GramCorrectionMode::Nvfp4GramOnlyLateSafety { .. } => CorrectionGram {
                values: self.nvfp4_gram(source, rows, cols, iter, stats)?,
                stale_reject_candidate: false,
                refresh: false,
            },
            GramCorrectionMode::Nvfp4GramAverage { samples } => CorrectionGram {
                values: self.averaged_nvfp4_gram(source, rows, cols, iter, samples, stats)?,
                stale_reject_candidate: false,
                refresh: false,
            },
            GramCorrectionMode::ExactPrefixThenNvfp4 { exact_steps } => {
                if iter < exact_steps {
                    CorrectionGram {
                        values: self.high_precision_gram(source, rows, cols, stats)?,
                        stale_reject_candidate: false,
                        refresh: true,
                    }
                } else {
                    CorrectionGram {
                        values: self.nvfp4_gram(source, rows, cols, iter, stats)?,
                        stale_reject_candidate: false,
                        refresh: false,
                    }
                }
            }
            GramCorrectionMode::ExactPrefixThenNvfp4Average {
                exact_steps,
                samples,
            } => {
                if iter < exact_steps {
                    CorrectionGram {
                        values: self.high_precision_gram(source, rows, cols, stats)?,
                        stale_reject_candidate: false,
                        refresh: true,
                    }
                } else {
                    CorrectionGram {
                        values: self
                            .averaged_nvfp4_gram(source, rows, cols, iter, samples, stats)?,
                        stale_reject_candidate: false,
                        refresh: false,
                    }
                }
            }
            GramCorrectionMode::Stale { period }
            | GramCorrectionMode::StaleReject { period }
            | GramCorrectionMode::StaleRejectSafety { period, .. } => self.stale_correction_gram(
                request,
                period <= 1 || iter % period == 0,
                1.0,
                rejects_stale_steps,
                stale_defect,
                stats,
            )?,
            GramCorrectionMode::StaleScaled { period, scale } => self.stale_correction_gram(
                request,
                period <= 1 || iter % period == 0,
                scale,
                false,
                stale_defect,
                stats,
            )?,
            GramCorrectionMode::ExactPrefixThenStale {
                exact_steps,
                period,
            }
            | GramCorrectionMode::ExactPrefixThenStaleReject {
                exact_steps,
                period,
            }
            | GramCorrectionMode::ExactPrefixThenStaleRejectSafety {
                exact_steps,
                period,
                ..
            }
            | GramCorrectionMode::ExactPrefixThenStaleRejectLateSafety {
                exact_steps,
                period,
                ..
            }
            | GramCorrectionMode::ExactPrefixThenStaleRejectSchedule {
                exact_steps,
                period,
                ..
            } => {
                if iter < exact_steps {
                    CorrectionGram {
                        values: self.high_precision_gram(source, rows, cols, stats)?,
                        stale_reject_candidate: false,
                        refresh: true,
                    }
                } else {
                    self.stale_correction_gram(
                        request,
                        iter == exact_steps || (iter - exact_steps) % period == 0,
                        1.0,
                        rejects_stale_steps,
                        stale_defect,
                        stats,
                    )?
                }
            }
            GramCorrectionMode::Adaptive {
                period,
                max_relative_defect,
            } => self.stale_correction_gram(
                request,
                period <= 1
                    || iter % period == 0
                    || stats.last_relative_defect > max_relative_defect,
                1.0,
                false,
                stale_defect,
                stats,
            )?,
        };
        Ok(gram)
    }

    fn stale_correction_gram(
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
        Ok(CorrectionGram {
            values: self.corrected_gram(
                source,
                rows,
                cols,
                iter,
                refresh,
                defect_scale,
                stale_defect,
                stats,
            )?,
            stale_reject_candidate: rejects_stale_steps && !refresh,
            refresh,
        })
    }

    fn high_precision_gram(
        &self,
        source: &[f32],
        rows: usize,
        cols: usize,
        stats: &mut CorrectionStats,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        stats.high_precision_gram_count += 1;
        self.f16_product(source, source, rows, rows, cols)
    }

    fn nvfp4_gram(
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
