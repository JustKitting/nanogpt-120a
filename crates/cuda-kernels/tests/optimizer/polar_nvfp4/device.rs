use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::f16_tc_matmul::{F16TcMatmulF32Args, F16TcMatmulModule};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tc_matmul::{Nvfp4TcMatmulArgs, Nvfp4TcMatmulModule};

use super::math::{combine_next, transpose};
use super::scratch::{Scratch, global_scale};

#[derive(Clone, Copy)]
pub enum GramCorrectionMode {
    HighPrecision,
    HighPrecisionSafety {
        coefficient_safety: f32,
    },
    Nvfp4GramOnly,
    Nvfp4GramOnlySafety {
        coefficient_safety: f32,
    },
    Nvfp4GramOnlySchedule {
        coefficient_safety: [f32; 8],
    },
    Nvfp4GramOnlyLateSafety {
        start_iter: usize,
        coefficient_safety: f32,
    },
    Nvfp4GramAverage {
        samples: usize,
    },
    ExactPrefixThenNvfp4 {
        exact_steps: usize,
    },
    ExactPrefixThenNvfp4Average {
        exact_steps: usize,
        samples: usize,
    },
    Stale {
        period: usize,
    },
    StaleReject {
        period: usize,
    },
    StaleRejectSafety {
        period: usize,
        coefficient_safety: f32,
    },
    StaleScaled {
        period: usize,
        scale: f32,
    },
    ExactPrefixThenStale {
        exact_steps: usize,
        period: usize,
    },
    ExactPrefixThenStaleReject {
        exact_steps: usize,
        period: usize,
    },
    ExactPrefixThenStaleRejectSafety {
        exact_steps: usize,
        period: usize,
        coefficient_safety: f32,
    },
    ExactPrefixThenStaleRejectLateSafety {
        exact_steps: usize,
        period: usize,
        start_iter: usize,
        coefficient_safety: f32,
    },
    ExactPrefixThenStaleRejectSchedule {
        exact_steps: usize,
        period: usize,
        coefficient_safety: [f32; 8],
    },
    Adaptive {
        period: usize,
        max_relative_defect: f32,
    },
}

impl GramCorrectionMode {
    fn coefficient_safety(self, iter: usize) -> f32 {
        match self {
            GramCorrectionMode::HighPrecisionSafety { coefficient_safety }
            | GramCorrectionMode::Nvfp4GramOnlySafety { coefficient_safety }
            | GramCorrectionMode::StaleRejectSafety {
                coefficient_safety, ..
            }
            | GramCorrectionMode::ExactPrefixThenStaleRejectSafety {
                coefficient_safety, ..
            } => coefficient_safety,
            GramCorrectionMode::Nvfp4GramOnlyLateSafety {
                start_iter,
                coefficient_safety,
            }
            | GramCorrectionMode::ExactPrefixThenStaleRejectLateSafety {
                start_iter,
                coefficient_safety,
                ..
            } => {
                if iter >= start_iter {
                    coefficient_safety
                } else {
                    1.0
                }
            }
            GramCorrectionMode::Nvfp4GramOnlySchedule { coefficient_safety }
            | GramCorrectionMode::ExactPrefixThenStaleRejectSchedule {
                coefficient_safety, ..
            } => coefficient_safety[iter.min(coefficient_safety.len() - 1)],
            _ => 1.0,
        }
    }
}

#[derive(Default)]
pub struct CorrectionStats {
    pub nvfp4_gram_count: usize,
    pub high_precision_gram_count: usize,
    pub max_relative_defect: f32,
    pub last_relative_defect: f32,
    pub rejected_stale_steps: usize,
}

pub struct Nvfp4Polar<'a> {
    stream: &'a CudaStream,
    f16: &'a F16TcMatmulModule,
    matmul: &'a Nvfp4TcMatmulModule,
    quant: &'a Nvfp4QuantModule,
}

impl<'a> Nvfp4Polar<'a> {
    pub fn new(
        stream: &'a CudaStream,
        f16: &'a F16TcMatmulModule,
        matmul: &'a Nvfp4TcMatmulModule,
        quant: &'a Nvfp4QuantModule,
    ) -> Self {
        Self {
            stream,
            f16,
            matmul,
            quant,
        }
    }

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
                    stats.high_precision_gram_count += 1;
                    self.f16_product(&source, &source, rows, rows, cols)?
                }
                GramCorrectionMode::Nvfp4GramOnly
                | GramCorrectionMode::Nvfp4GramOnlySafety { .. }
                | GramCorrectionMode::Nvfp4GramOnlySchedule { .. }
                | GramCorrectionMode::Nvfp4GramOnlyLateSafety { .. } => {
                    stats.nvfp4_gram_count += 1;
                    self.product(&source, &source, rows, rows, cols, iter, 0)?
                }
                GramCorrectionMode::Nvfp4GramAverage { samples } => {
                    self.averaged_nvfp4_gram(&source, rows, cols, iter, samples, &mut stats)?
                }
                GramCorrectionMode::ExactPrefixThenNvfp4 { exact_steps } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        self.f16_product(&source, &source, rows, rows, cols)?
                    } else {
                        stats.nvfp4_gram_count += 1;
                        self.product(&source, &source, rows, rows, cols, iter, 0)?
                    }
                }
                GramCorrectionMode::ExactPrefixThenNvfp4Average {
                    exact_steps,
                    samples,
                } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        self.f16_product(&source, &source, rows, rows, cols)?
                    } else {
                        self.averaged_nvfp4_gram(&source, rows, cols, iter, samples, &mut stats)?
                    }
                }
                GramCorrectionMode::Stale { period } => self.corrected_gram(
                    &source,
                    rows,
                    cols,
                    iter,
                    period <= 1 || iter % period == 0,
                    &mut stale_defect,
                    &mut stats,
                )?,
                GramCorrectionMode::StaleReject { period } => self.corrected_gram(
                    &source,
                    rows,
                    cols,
                    iter,
                    period <= 1 || iter % period == 0,
                    &mut stale_defect,
                    &mut stats,
                )?,
                GramCorrectionMode::StaleRejectSafety { period, .. } => self.corrected_gram(
                    &source,
                    rows,
                    cols,
                    iter,
                    period <= 1 || iter % period == 0,
                    &mut stale_defect,
                    &mut stats,
                )?,
                GramCorrectionMode::StaleScaled { period, scale } => self.corrected_gram_scaled(
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
                } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        self.f16_product(&source, &source, rows, rows, cols)?
                    } else {
                        self.corrected_gram(
                            &source,
                            rows,
                            cols,
                            iter,
                            iter == exact_steps || (iter - exact_steps) % period == 0,
                            &mut stale_defect,
                            &mut stats,
                        )?
                    }
                }
                GramCorrectionMode::ExactPrefixThenStaleReject {
                    exact_steps,
                    period,
                } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        self.f16_product(&source, &source, rows, rows, cols)?
                    } else {
                        self.corrected_gram(
                            &source,
                            rows,
                            cols,
                            iter,
                            iter == exact_steps || (iter - exact_steps) % period == 0,
                            &mut stale_defect,
                            &mut stats,
                        )?
                    }
                }
                GramCorrectionMode::ExactPrefixThenStaleRejectSafety {
                    exact_steps,
                    period,
                    ..
                } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        self.f16_product(&source, &source, rows, rows, cols)?
                    } else {
                        self.corrected_gram(
                            &source,
                            rows,
                            cols,
                            iter,
                            iter == exact_steps || (iter - exact_steps) % period == 0,
                            &mut stale_defect,
                            &mut stats,
                        )?
                    }
                }
                GramCorrectionMode::ExactPrefixThenStaleRejectLateSafety {
                    exact_steps,
                    period,
                    ..
                } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        self.f16_product(&source, &source, rows, rows, cols)?
                    } else {
                        self.corrected_gram(
                            &source,
                            rows,
                            cols,
                            iter,
                            iter == exact_steps || (iter - exact_steps) % period == 0,
                            &mut stale_defect,
                            &mut stats,
                        )?
                    }
                }
                GramCorrectionMode::ExactPrefixThenStaleRejectSchedule {
                    exact_steps,
                    period,
                    ..
                } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        self.f16_product(&source, &source, rows, rows, cols)?
                    } else {
                        self.corrected_gram(
                            &source,
                            rows,
                            cols,
                            iter,
                            iter == exact_steps || (iter - exact_steps) % period == 0,
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
            let (gram, stale_reject_candidate, refresh) = match mode {
                GramCorrectionMode::HighPrecision
                | GramCorrectionMode::HighPrecisionSafety { .. } => {
                    stats.high_precision_gram_count += 1;
                    (
                        self.f16_product(&source, &source, rows, rows, cols)?,
                        false,
                        true,
                    )
                }
                GramCorrectionMode::Nvfp4GramOnly
                | GramCorrectionMode::Nvfp4GramOnlySafety { .. }
                | GramCorrectionMode::Nvfp4GramOnlySchedule { .. }
                | GramCorrectionMode::Nvfp4GramOnlyLateSafety { .. } => {
                    stats.nvfp4_gram_count += 1;
                    (
                        self.product(&source, &source, rows, rows, cols, iter, 0)?,
                        false,
                        false,
                    )
                }
                GramCorrectionMode::Nvfp4GramAverage { samples } => (
                    self.averaged_nvfp4_gram(&source, rows, cols, iter, samples, &mut stats)?,
                    false,
                    false,
                ),
                GramCorrectionMode::ExactPrefixThenNvfp4 { exact_steps } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        (
                            self.f16_product(&source, &source, rows, rows, cols)?,
                            false,
                            true,
                        )
                    } else {
                        stats.nvfp4_gram_count += 1;
                        (
                            self.product(&source, &source, rows, rows, cols, iter, 0)?,
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
                        stats.high_precision_gram_count += 1;
                        (
                            self.f16_product(&source, &source, rows, rows, cols)?,
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
                GramCorrectionMode::Stale { period } => {
                    let refresh = period <= 1 || iter % period == 0;
                    (
                        self.corrected_gram(
                            &source,
                            rows,
                            cols,
                            iter,
                            refresh,
                            &mut stale_defect,
                            &mut stats,
                        )?,
                        false,
                        refresh,
                    )
                }
                GramCorrectionMode::StaleReject { period } => {
                    let refresh = period <= 1 || iter % period == 0;
                    (
                        self.corrected_gram(
                            &source,
                            rows,
                            cols,
                            iter,
                            refresh,
                            &mut stale_defect,
                            &mut stats,
                        )?,
                        !refresh,
                        refresh,
                    )
                }
                GramCorrectionMode::StaleRejectSafety { period, .. } => {
                    let refresh = period <= 1 || iter % period == 0;
                    (
                        self.corrected_gram(
                            &source,
                            rows,
                            cols,
                            iter,
                            refresh,
                            &mut stale_defect,
                            &mut stats,
                        )?,
                        !refresh,
                        refresh,
                    )
                }
                GramCorrectionMode::StaleScaled { period, scale } => {
                    let refresh = period <= 1 || iter % period == 0;
                    (
                        self.corrected_gram_scaled(
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
                } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        (
                            self.f16_product(&source, &source, rows, rows, cols)?,
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
                                &mut stale_defect,
                                &mut stats,
                            )?,
                            false,
                            refresh,
                        )
                    }
                }
                GramCorrectionMode::ExactPrefixThenStaleReject {
                    exact_steps,
                    period,
                } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        (
                            self.f16_product(&source, &source, rows, rows, cols)?,
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
                                &mut stale_defect,
                                &mut stats,
                            )?,
                            !refresh,
                            refresh,
                        )
                    }
                }
                GramCorrectionMode::ExactPrefixThenStaleRejectSafety {
                    exact_steps,
                    period,
                    ..
                } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        (
                            self.f16_product(&source, &source, rows, rows, cols)?,
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
                                &mut stale_defect,
                                &mut stats,
                            )?,
                            !refresh,
                            refresh,
                        )
                    }
                }
                GramCorrectionMode::ExactPrefixThenStaleRejectLateSafety {
                    exact_steps,
                    period,
                    ..
                } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        (
                            self.f16_product(&source, &source, rows, rows, cols)?,
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
                                &mut stale_defect,
                                &mut stats,
                            )?,
                            !refresh,
                            refresh,
                        )
                    }
                }
                GramCorrectionMode::ExactPrefixThenStaleRejectSchedule {
                    exact_steps,
                    period,
                    ..
                } => {
                    if iter < exact_steps {
                        stats.high_precision_gram_count += 1;
                        (
                            self.f16_product(&source, &source, rows, rows, cols)?,
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
                                &mut stale_defect,
                                &mut stats,
                            )?,
                            !refresh,
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

    #[allow(clippy::too_many_arguments)]
    fn corrected_gram_scaled(
        &self,
        source: &[f32],
        rows: usize,
        cols: usize,
        iter: usize,
        refresh: bool,
        defect_scale: f32,
        stale_defect: &mut [f32],
        stats: &mut CorrectionStats,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        stats.nvfp4_gram_count += 1;
        let gram_q = self.product(source, source, rows, rows, cols, iter, 0)?;
        if refresh {
            stats.high_precision_gram_count += 1;
            let gram_hi = self.f16_product(source, source, rows, rows, cols)?;
            for ((defect, q), hi) in stale_defect.iter_mut().zip(&gram_q).zip(&gram_hi) {
                *defect = q - hi;
            }
            stats.last_relative_defect = relative_l2(&gram_q, &gram_hi);
            stats.max_relative_defect = stats.max_relative_defect.max(stats.last_relative_defect);
            Ok(gram_hi)
        } else {
            Ok(gram_q
                .iter()
                .zip(stale_defect)
                .map(|(q, defect)| defect_scale.mul_add(-*defect, *q))
                .collect())
        }
    }

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

    fn corrected_gram(
        &self,
        source: &[f32],
        rows: usize,
        cols: usize,
        iter: usize,
        refresh: bool,
        stale_defect: &mut [f32],
        stats: &mut CorrectionStats,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        stats.nvfp4_gram_count += 1;
        let gram_q = self.product(source, source, rows, rows, cols, iter, 0)?;
        if refresh {
            stats.high_precision_gram_count += 1;
            let gram_hi = self.f16_product(source, source, rows, rows, cols)?;
            for ((defect, q), hi) in stale_defect.iter_mut().zip(&gram_q).zip(&gram_hi) {
                *defect = q - hi;
            }
            stats.last_relative_defect = relative_l2(&gram_q, &gram_hi);
            stats.max_relative_defect = stats.max_relative_defect.max(stats.last_relative_defect);
            Ok(gram_hi)
        } else {
            Ok(gram_q
                .iter()
                .zip(stale_defect)
                .map(|(q, defect)| *q - *defect)
                .collect())
        }
    }

    fn step_from_gram(
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

    fn gram_form_step_from_gram(
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

    fn averaged_nvfp4_gram(
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

    pub fn product(
        &self,
        a: &[f32],
        b_t: &[f32],
        m: usize,
        n: usize,
        k: usize,
        iter: usize,
        stage: u32,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let a_dev = DeviceBuffer::from_host(self.stream, a)?;
        let b_t_dev = DeviceBuffer::from_host(self.stream, b_t)?;
        let mut out = DeviceBuffer::<f32>::zeroed(self.stream, m * n)?;
        let mut scratch = Scratch::new(self.stream, m, n, k, global_scale(a), global_scale(b_t))?;

        self.matmul.matmul_ms_eden(Nvfp4TcMatmulArgs {
            stream: self.stream,
            quant_module: self.quant,
            a: &a_dev,
            b_t: &b_t_dev,
            out: &mut out,
            scratch: scratch.args(),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            sign_seed: seed(iter, stage, 0),
            scale_seed: seed(iter, stage, 1),
        })?;

        Ok(out.to_host_vec(self.stream)?)
    }

    fn f16_product(
        &self,
        a: &[f32],
        b_t: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let a_dev = DeviceBuffer::from_host(self.stream, a)?;
        let b_t_dev = DeviceBuffer::from_host(self.stream, b_t)?;
        let mut out = DeviceBuffer::<f32>::zeroed(self.stream, m * n)?;

        self.f16.batched_matmul_f32_input(F16TcMatmulF32Args {
            stream: self.stream,
            a: &a_dev,
            b_t: &b_t_dev,
            out: &mut out,
            batch_count: 1,
            m: m as u32,
            n: n as u32,
            k: k as u32,
        })?;

        Ok(out.to_host_vec(self.stream)?)
    }
}

fn relative_l2(actual: &[f32], expected: &[f32]) -> f32 {
    let (err, norm) =
        actual
            .iter()
            .zip(expected)
            .fold((0.0_f32, 0.0_f32), |(err, norm), (actual, expected)| {
                let diff = actual - expected;
                (diff.mul_add(diff, err), expected.mul_add(*expected, norm))
            });
    (err / norm).sqrt()
}

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

fn seed(iter: usize, stage: u32, stream: u32) -> u32 {
    0x9e37_79b9_u32
        .wrapping_mul((iter as u32).wrapping_add(1))
        .wrapping_add(stage.wrapping_mul(0x85eb_ca6b))
        .wrapping_add(stream.wrapping_mul(0xc2b2_ae35))
}
