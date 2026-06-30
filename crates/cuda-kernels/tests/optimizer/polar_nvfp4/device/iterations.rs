use std::error::Error;

use super::{CorrectionStats, GramCorrectionMode, Nvfp4Polar, row_orthogonality_residual};

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
            let gram = match mode {
                GramCorrectionMode::HighPrecision
                | GramCorrectionMode::HighPrecisionSafety { .. } => {
                    self.high_precision_gram(&source, rows, cols, &mut stats)?
                }
                GramCorrectionMode::Nvfp4GramOnly
                | GramCorrectionMode::Nvfp4GramOnlySafety { .. }
                | GramCorrectionMode::Nvfp4GramOnlySchedule { .. }
                | GramCorrectionMode::Nvfp4GramOnlyLateSafety { .. } => {
                    self.nvfp4_gram(&source, rows, cols, iter, &mut stats)?
                }
                GramCorrectionMode::Nvfp4GramAverage { samples } => {
                    self.averaged_nvfp4_gram(&source, rows, cols, iter, samples, &mut stats)?
                }
                GramCorrectionMode::ExactPrefixThenNvfp4 { exact_steps } => {
                    if iter < exact_steps {
                        self.high_precision_gram(&source, rows, cols, &mut stats)?
                    } else {
                        self.nvfp4_gram(&source, rows, cols, iter, &mut stats)?
                    }
                }
                GramCorrectionMode::ExactPrefixThenNvfp4Average {
                    exact_steps,
                    samples,
                } => {
                    if iter < exact_steps {
                        self.high_precision_gram(&source, rows, cols, &mut stats)?
                    } else {
                        self.averaged_nvfp4_gram(&source, rows, cols, iter, samples, &mut stats)?
                    }
                }
                GramCorrectionMode::Stale { period }
                | GramCorrectionMode::StaleReject { period }
                | GramCorrectionMode::StaleRejectSafety { period, .. } => self.corrected_gram(
                    &source,
                    rows,
                    cols,
                    iter,
                    period <= 1 || iter % period == 0,
                    1.0,
                    &mut stale_defect,
                    &mut stats,
                )?,
                GramCorrectionMode::StaleScaled { period, scale } => self.corrected_gram(
                    &source,
                    rows,
                    cols,
                    iter,
                    period <= 1 || iter % period == 0,
                    scale,
                    &mut stale_defect,
                    &mut stats,
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
                        self.high_precision_gram(&source, rows, cols, &mut stats)?
                    } else {
                        self.corrected_gram(
                            &source,
                            rows,
                            cols,
                            iter,
                            iter == exact_steps || (iter - exact_steps) % period == 0,
                            1.0,
                            &mut stale_defect,
                            &mut stats,
                        )?
                    }
                }
                GramCorrectionMode::Adaptive {
                    period,
                    max_relative_defect,
                } => self.corrected_gram(
                    &source,
                    rows,
                    cols,
                    iter,
                    period <= 1
                        || iter % period == 0
                        || stats.last_relative_defect > max_relative_defect,
                    1.0,
                    &mut stale_defect,
                    &mut stats,
                )?,
            };
            source = self.step_from_gram(&source, &gram, rows, cols, iter)?;
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
            let rejects_stale_steps = mode.rejects_stale_steps();
            let (gram, stale_reject_candidate, refresh) = match mode {
                GramCorrectionMode::HighPrecision
                | GramCorrectionMode::HighPrecisionSafety { .. } => (
                    self.high_precision_gram(&source, rows, cols, &mut stats)?,
                    false,
                    true,
                ),
                GramCorrectionMode::Nvfp4GramOnly
                | GramCorrectionMode::Nvfp4GramOnlySafety { .. }
                | GramCorrectionMode::Nvfp4GramOnlySchedule { .. }
                | GramCorrectionMode::Nvfp4GramOnlyLateSafety { .. } => (
                    self.nvfp4_gram(&source, rows, cols, iter, &mut stats)?,
                    false,
                    false,
                ),
                GramCorrectionMode::Nvfp4GramAverage { samples } => (
                    self.averaged_nvfp4_gram(&source, rows, cols, iter, samples, &mut stats)?,
                    false,
                    false,
                ),
                GramCorrectionMode::ExactPrefixThenNvfp4 { exact_steps } => {
                    if iter < exact_steps {
                        (
                            self.high_precision_gram(&source, rows, cols, &mut stats)?,
                            false,
                            true,
                        )
                    } else {
                        (
                            self.nvfp4_gram(&source, rows, cols, iter, &mut stats)?,
                            false,
                            false,
                        )
                    }
                }
                GramCorrectionMode::ExactPrefixThenNvfp4Average {
                    exact_steps,
                    samples,
                } => {
                    if iter < exact_steps {
                        (
                            self.high_precision_gram(&source, rows, cols, &mut stats)?,
                            false,
                            true,
                        )
                    } else {
                        (
                            self.averaged_nvfp4_gram(
                                &source, rows, cols, iter, samples, &mut stats,
                            )?,
                            false,
                            false,
                        )
                    }
                }
                GramCorrectionMode::Stale { period }
                | GramCorrectionMode::StaleReject { period }
                | GramCorrectionMode::StaleRejectSafety { period, .. } => {
                    let refresh = period <= 1 || iter % period == 0;
                    (
                        self.corrected_gram(
                            &source,
                            rows,
                            cols,
                            iter,
                            refresh,
                            1.0,
                            &mut stale_defect,
                            &mut stats,
                        )?,
                        rejects_stale_steps && !refresh,
                        refresh,
                    )
                }
                GramCorrectionMode::StaleScaled { period, scale } => {
                    let refresh = period <= 1 || iter % period == 0;
                    (
                        self.corrected_gram(
                            &source,
                            rows,
                            cols,
                            iter,
                            refresh,
                            scale,
                            &mut stale_defect,
                            &mut stats,
                        )?,
                        false,
                        refresh,
                    )
                }
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
                        (
                            self.high_precision_gram(&source, rows, cols, &mut stats)?,
                            false,
                            true,
                        )
                    } else {
                        let refresh = iter == exact_steps || (iter - exact_steps) % period == 0;
                        (
                            self.corrected_gram(
                                &source,
                                rows,
                                cols,
                                iter,
                                refresh,
                                1.0,
                                &mut stale_defect,
                                &mut stats,
                            )?,
                            rejects_stale_steps && !refresh,
                            refresh,
                        )
                    }
                }
                GramCorrectionMode::Adaptive {
                    period,
                    max_relative_defect,
                } => {
                    let refresh = period <= 1
                        || iter % period == 0
                        || stats.last_relative_defect > max_relative_defect;
                    (
                        self.corrected_gram(
                            &source,
                            rows,
                            cols,
                            iter,
                            refresh,
                            1.0,
                            &mut stale_defect,
                            &mut stats,
                        )?,
                        false,
                        refresh,
                    )
                }
            };

            let candidate = self.gram_form_step_from_gram(
                &source,
                &gram,
                rows,
                cols,
                iter,
                coefficient_safety,
            )?;
            let candidate_residual = row_orthogonality_residual(&candidate, rows, cols);
            if stale_reject_candidate && candidate_residual > residual {
                stats.rejected_stale_steps += 1;
                continue;
            }

            source = candidate;
            residual = candidate_residual;
            if refresh {
                residual = row_orthogonality_residual(&source, rows, cols);
            }
        }

        Ok((source, stats))
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
