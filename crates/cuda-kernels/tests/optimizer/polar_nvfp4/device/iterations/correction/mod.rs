use std::error::Error;

use super::super::{CorrectionStats, GramCorrectionMode, Nvfp4Polar};

mod gram;
mod types;

pub(super) use types::{CorrectionGram, GramRequest};

impl<'a> Nvfp4Polar<'a> {
    pub(super) fn correction_gram(
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
                CorrectionGram::refreshed(self.high_precision_gram(source, rows, cols, stats)?)
            }
            GramCorrectionMode::Nvfp4GramOnly
            | GramCorrectionMode::Nvfp4GramOnlySafety { .. }
            | GramCorrectionMode::Nvfp4GramOnlySchedule { .. }
            | GramCorrectionMode::Nvfp4GramOnlyLateSafety { .. } => {
                CorrectionGram::approximate(self.nvfp4_gram(source, rows, cols, iter, stats)?)
            }
            GramCorrectionMode::Nvfp4GramAverage { samples } => CorrectionGram::approximate(
                self.averaged_nvfp4_gram(source, rows, cols, iter, samples, stats)?,
            ),
            GramCorrectionMode::ExactPrefixThenNvfp4 { exact_steps } => {
                if iter < exact_steps {
                    CorrectionGram::refreshed(self.high_precision_gram(source, rows, cols, stats)?)
                } else {
                    CorrectionGram::approximate(self.nvfp4_gram(source, rows, cols, iter, stats)?)
                }
            }
            GramCorrectionMode::ExactPrefixThenNvfp4Average {
                exact_steps,
                samples,
            } => {
                if iter < exact_steps {
                    CorrectionGram::refreshed(self.high_precision_gram(source, rows, cols, stats)?)
                } else {
                    CorrectionGram::approximate(
                        self.averaged_nvfp4_gram(source, rows, cols, iter, samples, stats)?,
                    )
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
                    CorrectionGram::refreshed(self.high_precision_gram(source, rows, cols, stats)?)
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
}
