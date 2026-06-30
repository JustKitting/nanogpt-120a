use std::error::Error;

use cuda_core::CudaContext;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tc_matmul::Nvfp4TcMatmulModule;

use crate::common;

#[path = "polar_nvfp4/device.rs"]
mod device;
#[path = "polar_nvfp4/math.rs"]
mod math;
#[path = "polar_nvfp4/schedule.rs"]
mod schedule;
#[path = "polar_nvfp4/scratch.rs"]
mod scratch;

use schedule::{
    ScheduleResult, corrected_schedule_candidates, evaluate_mode, evaluate_schedule,
    evaluate_stale_reject_schedule, production_shape_modes, report_schedule, schedule_name,
    search_best_corrected_schedule, search_best_raw_schedule, seed_safety_schedules,
};

const ROWS: usize = 32;
const COLS: usize = 64;
const MAX_ITERATIONS: usize = 8;
const PRODUCTION_ITERATIONS: usize = 5;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_rht_polar_estimator_reports_update_error() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
    let f16 = F16TcMatmulModule::from_module(ptx.clone())?;
    let matmul = Nvfp4TcMatmulModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;
    let polar = device::Nvfp4Polar::new(&stream, &f16, &matmul, &quant);

    let source = math::normalized_source(&math::gradient(ROWS, COLS), ROWS, COLS);
    let gram_expected = math::matmul_f16_leaf(&source, &source, ROWS, ROWS, COLS);
    let gram_actual = polar.product(&source, &source, ROWS, ROWS, COLS, 0, 0)?;
    println!(
        "nvfp4_rht_polar first_gram cosine={:.8} rel_l2={:.8e} max_abs={:.8e}",
        math::cosine(&gram_actual, &gram_expected),
        math::relative_l2(&gram_actual, &gram_expected),
        math::max_abs_error(&gram_actual, &gram_expected)
    );
    let source_t = math::transpose(&source, ROWS, COLS);
    let ax_expected = math::matmul_f16_leaf(&gram_expected, &source_t, ROWS, COLS, ROWS);
    let ax_actual = polar.product(&gram_expected, &source_t, ROWS, COLS, ROWS, 0, 1)?;
    println!(
        "nvfp4_rht_polar first_ax_from_expected_gram cosine={:.8} rel_l2={:.8e} max_abs={:.8e}",
        math::cosine(&ax_actual, &ax_expected),
        math::relative_l2(&ax_actual, &ax_expected),
        math::max_abs_error(&ax_actual, &ax_expected)
    );
    let ax_t = math::transpose(&ax_expected, ROWS, COLS);
    let aax_expected = math::matmul_f16_leaf(&gram_expected, &ax_t, ROWS, COLS, ROWS);
    let aax_actual = polar.product(&gram_expected, &ax_t, ROWS, COLS, ROWS, 0, 2)?;
    println!(
        "nvfp4_rht_polar first_aax_from_expected_inputs cosine={:.8} rel_l2={:.8e} max_abs={:.8e}",
        math::cosine(&aax_actual, &aax_expected),
        math::relative_l2(&aax_actual, &aax_expected),
        math::max_abs_error(&aax_actual, &aax_expected)
    );

    for iterations in 1..=MAX_ITERATIONS {
        let expected = math::polar_iterations_f16_leaf(source.clone(), ROWS, COLS, iterations);
        let actual = polar.iterations(source.clone(), ROWS, COLS, iterations)?;
        let cosine = math::cosine(&actual, &expected);
        let rel_l2 = math::relative_l2(&actual, &expected);
        let max_abs = math::max_abs_error(&actual, &expected);
        println!(
            "nvfp4_rht_polar iterations={iterations} cosine={cosine:.8} rel_l2={rel_l2:.8e} max_abs={max_abs:.8e}"
        );
    }

    let expected = math::polar_iterations_f16_leaf(source.clone(), ROWS, COLS, 5);
    for fp4_prefix in 1..=3 {
        let mut actual = source.clone();
        for iter in 0..5 {
            actual = if iter < fp4_prefix {
                polar.step(&actual, ROWS, COLS, iter)?
            } else {
                math::polar_step_f16_leaf(&actual, ROWS, COLS, iter)
            };
        }
        println!(
            "nvfp4_rht_polar hybrid_fp4_prefix={fp4_prefix} cosine={:.8} rel_l2={:.8e} max_abs={:.8e}",
            math::cosine(&actual, &expected),
            math::relative_l2(&actual, &expected),
            math::max_abs_error(&actual, &expected)
        );
    }

    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_gram_correction_variants_report_update_error() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
    let f16 = F16TcMatmulModule::from_module(ptx.clone())?;
    let matmul = Nvfp4TcMatmulModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;
    let polar = device::Nvfp4Polar::new(&stream, &f16, &matmul, &quant);
    let source = math::normalized_source(&math::gradient(ROWS, COLS), ROWS, COLS);

    for iterations in 1..=MAX_ITERATIONS {
        let expected = math::polar_iterations_f16_leaf(source.clone(), ROWS, COLS, iterations);
        for (name, mode) in [
            ("high_precision", device::GramCorrectionMode::HighPrecision),
            ("nvfp4_gram_only", device::GramCorrectionMode::Nvfp4GramOnly),
            (
                "nvfp4_avg2",
                device::GramCorrectionMode::Nvfp4GramAverage { samples: 2 },
            ),
            (
                "nvfp4_avg4",
                device::GramCorrectionMode::Nvfp4GramAverage { samples: 4 },
            ),
            (
                "nvfp4_avg8",
                device::GramCorrectionMode::Nvfp4GramAverage { samples: 8 },
            ),
            (
                "prefix2_nvfp4",
                device::GramCorrectionMode::ExactPrefixThenNvfp4 { exact_steps: 2 },
            ),
            (
                "prefix3_nvfp4",
                device::GramCorrectionMode::ExactPrefixThenNvfp4 { exact_steps: 3 },
            ),
            (
                "prefix2_avg4",
                device::GramCorrectionMode::ExactPrefixThenNvfp4Average {
                    exact_steps: 2,
                    samples: 4,
                },
            ),
            (
                "prefix3_avg4",
                device::GramCorrectionMode::ExactPrefixThenNvfp4Average {
                    exact_steps: 3,
                    samples: 4,
                },
            ),
            (
                "stale_period1",
                device::GramCorrectionMode::Stale { period: 1 },
            ),
            (
                "stale_period2",
                device::GramCorrectionMode::Stale { period: 2 },
            ),
            (
                "stale_period3",
                device::GramCorrectionMode::Stale { period: 3 },
            ),
            (
                "prefix2_stale2",
                device::GramCorrectionMode::ExactPrefixThenStale {
                    exact_steps: 2,
                    period: 2,
                },
            ),
            (
                "prefix3_stale2",
                device::GramCorrectionMode::ExactPrefixThenStale {
                    exact_steps: 3,
                    period: 2,
                },
            ),
            (
                "prefix4_stale2",
                device::GramCorrectionMode::ExactPrefixThenStale {
                    exact_steps: 4,
                    period: 2,
                },
            ),
            (
                "adaptive_p2_e03",
                device::GramCorrectionMode::Adaptive {
                    period: 2,
                    max_relative_defect: 3.0e-2,
                },
            ),
            (
                "adaptive_p2_e05",
                device::GramCorrectionMode::Adaptive {
                    period: 2,
                    max_relative_defect: 5.0e-2,
                },
            ),
            (
                "adaptive_p3_e05",
                device::GramCorrectionMode::Adaptive {
                    period: 3,
                    max_relative_defect: 5.0e-2,
                },
            ),
        ] {
            let (actual, stats) =
                polar.gram_corrected_iterations(source.clone(), ROWS, COLS, iterations, mode)?;
            let finite = actual.iter().all(|value| value.is_finite());
            println!(
                "nvfp4_gram_correction mode={name} iterations={iterations} finite={finite} cosine={:.8} rel_l2={:.8e} max_abs={:.8e} nvfp4_grams={} hi_grams={} max_defect={:.8e} last_defect={:.8e}",
                math::cosine(&actual, &expected),
                math::relative_l2(&actual, &expected),
                math::max_abs_error(&actual, &expected),
                stats.nvfp4_gram_count,
                stats.high_precision_gram_count,
                stats.max_relative_defect,
                stats.last_relative_defect,
            );
        }
    }

    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_gram_form_correction_variants_report_update_error() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
    let f16 = F16TcMatmulModule::from_module(ptx.clone())?;
    let matmul = Nvfp4TcMatmulModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;
    let polar = device::Nvfp4Polar::new(&stream, &f16, &matmul, &quant);
    let source = math::normalized_source(&math::gradient(ROWS, COLS), ROWS, COLS);

    for iterations in 1..=MAX_ITERATIONS {
        let expected =
            math::gram_form_polar_iterations_f16_leaf(source.clone(), ROWS, COLS, iterations);
        for (name, mode) in [
            ("high_precision", device::GramCorrectionMode::HighPrecision),
            (
                "high_precision_extra_safety101",
                device::GramCorrectionMode::HighPrecisionSafety {
                    coefficient_safety: 1.01,
                },
            ),
            (
                "high_precision_extra_safety103",
                device::GramCorrectionMode::HighPrecisionSafety {
                    coefficient_safety: 1.03,
                },
            ),
            (
                "high_precision_extra_safety104",
                device::GramCorrectionMode::HighPrecisionSafety {
                    coefficient_safety: 1.04,
                },
            ),
            (
                "high_precision_extra_safety1045",
                device::GramCorrectionMode::HighPrecisionSafety {
                    coefficient_safety: 1.045,
                },
            ),
            (
                "high_precision_extra_safety105",
                device::GramCorrectionMode::HighPrecisionSafety {
                    coefficient_safety: 1.05,
                },
            ),
            ("nvfp4_gram_only", device::GramCorrectionMode::Nvfp4GramOnly),
            (
                "nvfp4_gram_only_extra_safety101",
                device::GramCorrectionMode::Nvfp4GramOnlySafety {
                    coefficient_safety: 1.01,
                },
            ),
            (
                "nvfp4_gram_only_extra_safety103",
                device::GramCorrectionMode::Nvfp4GramOnlySafety {
                    coefficient_safety: 1.03,
                },
            ),
            (
                "nvfp4_gram_only_extra_safety104",
                device::GramCorrectionMode::Nvfp4GramOnlySafety {
                    coefficient_safety: 1.04,
                },
            ),
            (
                "nvfp4_gram_only_extra_safety1045",
                device::GramCorrectionMode::Nvfp4GramOnlySafety {
                    coefficient_safety: 1.045,
                },
            ),
            (
                "nvfp4_gram_only_extra_safety105",
                device::GramCorrectionMode::Nvfp4GramOnlySafety {
                    coefficient_safety: 1.05,
                },
            ),
            (
                "nvfp4_gram_only_late4_safety1045",
                device::GramCorrectionMode::Nvfp4GramOnlyLateSafety {
                    start_iter: 4,
                    coefficient_safety: 1.045,
                },
            ),
            (
                "nvfp4_gram_only_late4_safety105",
                device::GramCorrectionMode::Nvfp4GramOnlyLateSafety {
                    start_iter: 4,
                    coefficient_safety: 1.05,
                },
            ),
            (
                "nvfp4_gram_only_late3_safety1045",
                device::GramCorrectionMode::Nvfp4GramOnlyLateSafety {
                    start_iter: 3,
                    coefficient_safety: 1.045,
                },
            ),
            (
                "nvfp4_gram_only_late3_safety105",
                device::GramCorrectionMode::Nvfp4GramOnlyLateSafety {
                    start_iter: 3,
                    coefficient_safety: 1.05,
                },
            ),
            (
                "nvfp4_gram_only_late2_safety1045",
                device::GramCorrectionMode::Nvfp4GramOnlyLateSafety {
                    start_iter: 2,
                    coefficient_safety: 1.045,
                },
            ),
            (
                "nvfp4_gram_only_late5_safety105",
                device::GramCorrectionMode::Nvfp4GramOnlyLateSafety {
                    start_iter: 5,
                    coefficient_safety: 1.05,
                },
            ),
            (
                "nvfp4_avg4",
                device::GramCorrectionMode::Nvfp4GramAverage { samples: 4 },
            ),
            (
                "stale_period2",
                device::GramCorrectionMode::Stale { period: 2 },
            ),
            (
                "stale_reject_period2",
                device::GramCorrectionMode::StaleReject { period: 2 },
            ),
            (
                "stale_reject_p2_extra_safety101",
                device::GramCorrectionMode::StaleRejectSafety {
                    period: 2,
                    coefficient_safety: 1.01,
                },
            ),
            (
                "stale_reject_p2_extra_safety103",
                device::GramCorrectionMode::StaleRejectSafety {
                    period: 2,
                    coefficient_safety: 1.03,
                },
            ),
            (
                "stale_reject_p2_extra_safety105",
                device::GramCorrectionMode::StaleRejectSafety {
                    period: 2,
                    coefficient_safety: 1.05,
                },
            ),
            (
                "stale_scaled25_period2",
                device::GramCorrectionMode::StaleScaled {
                    period: 2,
                    scale: 0.25,
                },
            ),
            (
                "stale_scaled50_period2",
                device::GramCorrectionMode::StaleScaled {
                    period: 2,
                    scale: 0.50,
                },
            ),
            (
                "stale_scaled75_period2",
                device::GramCorrectionMode::StaleScaled {
                    period: 2,
                    scale: 0.75,
                },
            ),
            (
                "prefix2_stale_reject2",
                device::GramCorrectionMode::ExactPrefixThenStaleReject {
                    exact_steps: 2,
                    period: 2,
                },
            ),
            (
                "prefix3_stale_reject2",
                device::GramCorrectionMode::ExactPrefixThenStaleReject {
                    exact_steps: 3,
                    period: 2,
                },
            ),
            (
                "prefix3_stale_reject2_extra_safety101",
                device::GramCorrectionMode::ExactPrefixThenStaleRejectSafety {
                    exact_steps: 3,
                    period: 2,
                    coefficient_safety: 1.01,
                },
            ),
            (
                "prefix3_stale_reject2_extra_safety103",
                device::GramCorrectionMode::ExactPrefixThenStaleRejectSafety {
                    exact_steps: 3,
                    period: 2,
                    coefficient_safety: 1.03,
                },
            ),
            (
                "prefix3_stale_reject2_extra_safety105",
                device::GramCorrectionMode::ExactPrefixThenStaleRejectSafety {
                    exact_steps: 3,
                    period: 2,
                    coefficient_safety: 1.05,
                },
            ),
            (
                "prefix3_stale_reject2_late4_safety103",
                device::GramCorrectionMode::ExactPrefixThenStaleRejectLateSafety {
                    exact_steps: 3,
                    period: 2,
                    start_iter: 4,
                    coefficient_safety: 1.03,
                },
            ),
            (
                "prefix3_stale_reject2_late5_safety103",
                device::GramCorrectionMode::ExactPrefixThenStaleRejectLateSafety {
                    exact_steps: 3,
                    period: 2,
                    start_iter: 5,
                    coefficient_safety: 1.03,
                },
            ),
        ] {
            let (actual, stats) = polar.gram_form_corrected_iterations(
                source.clone(),
                ROWS,
                COLS,
                iterations,
                mode,
            )?;
            let finite = actual.iter().all(|value| value.is_finite());
            println!(
                "nvfp4_gram_form_correction mode={name} iterations={iterations} finite={finite} cosine={:.8} rel_l2={:.8e} max_abs={:.8e} nvfp4_grams={} hi_grams={} rejected={} max_defect={:.8e} last_defect={:.8e}",
                math::cosine(&actual, &expected),
                math::relative_l2(&actual, &expected),
                math::max_abs_error(&actual, &expected),
                stats.nvfp4_gram_count,
                stats.high_precision_gram_count,
                stats.rejected_stale_steps,
                stats.max_relative_defect,
                stats.last_relative_defect,
            );
        }
    }

    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_gram_form_safety_schedule_search() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
    let f16 = F16TcMatmulModule::from_module(ptx.clone())?;
    let matmul = Nvfp4TcMatmulModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;
    let polar = device::Nvfp4Polar::new(&stream, &f16, &matmul, &quant);
    let source = math::normalized_source(&math::gradient(ROWS, COLS), ROWS, COLS);
    let expected =
        math::gram_form_polar_iterations_f16_leaf(source.clone(), ROWS, COLS, MAX_ITERATIONS);

    let mut best = ScheduleResult::missing();
    for schedule in seed_safety_schedules() {
        let result = evaluate_schedule(
            &polar,
            &source,
            &expected,
            &format!("seed_{}", schedule_name(&schedule)),
            schedule,
            ROWS,
            COLS,
            MAX_ITERATIONS,
            false,
        )?;
        report_schedule(&result);
        if result.is_better_than(&best) {
            best = result;
        }
    }

    let mut greedy = best.schedule;
    let values = [1.0, 1.01, 1.02, 1.03, 1.04, 1.045, 1.05, 1.06, 1.08];
    for pass in 0..2 {
        for iter in 0..MAX_ITERATIONS {
            let mut local_best = best.clone();
            for value in values {
                let mut candidate = greedy;
                candidate[iter] = value;
                let result = evaluate_schedule(
                    &polar,
                    &source,
                    &expected,
                    &format!("greedy_pass{pass}_iter{iter}_{value:.3}"),
                    candidate,
                    ROWS,
                    COLS,
                    MAX_ITERATIONS,
                    false,
                )?;
                if result.is_better_than(&local_best) {
                    local_best = result;
                }
            }
            if local_best.is_better_than(&best) {
                report_schedule(&local_best);
                best = local_best;
                greedy = best.schedule;
            }
        }
    }

    println!(
        "nvfp4_gram_form_schedule_search_best kind=nvfp4_only name={} finite={} cosine={:.8} rel_l2={:.8e} max_abs={:.8e} nvfp4_grams={} hi_grams={} rejected={} schedule={}",
        best.name,
        best.finite,
        best.cosine,
        best.rel_l2,
        best.max_abs,
        best.nvfp4_grams,
        best.hi_grams,
        best.rejected,
        schedule_name(&best.schedule)
    );

    let mut best_corrected = ScheduleResult::missing();
    for (exact_steps, period) in [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 5),
    ] {
        for schedule in corrected_schedule_candidates(best.schedule) {
            let result = evaluate_stale_reject_schedule(
                &polar,
                &source,
                &expected,
                exact_steps,
                period,
                schedule,
                ROWS,
                COLS,
                MAX_ITERATIONS,
            )?;
            report_schedule(&result);
            if result.is_better_than(&best_corrected) {
                best_corrected = result;
            }
        }
    }
    println!(
        "nvfp4_gram_form_schedule_search_best kind=stale_reject name={} finite={} cosine={:.8} rel_l2={:.8e} max_abs={:.8e} nvfp4_grams={} hi_grams={} rejected={} schedule={}",
        best_corrected.name,
        best_corrected.finite,
        best_corrected.cosine,
        best_corrected.rel_l2,
        best_corrected.max_abs,
        best_corrected.nvfp4_grams,
        best_corrected.hi_grams,
        best_corrected.rejected,
        schedule_name(&best_corrected.schedule)
    );

    assert!(
        best.finite,
        "at least one static NVFP4 safety schedule must survive"
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_gram_form_ratio_schedule_search() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
    let f16 = F16TcMatmulModule::from_module(ptx.clone())?;
    let matmul = Nvfp4TcMatmulModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;
    let polar = device::Nvfp4Polar::new(&stream, &f16, &matmul, &quant);

    for (name, rows, cols) in [
        ("square_ratio", 128, 128),
        ("qkv_ratio", 128, 384),
        ("mlp_ratio", 128, 512),
    ] {
        let source = math::normalized_source(&math::gradient(rows, cols), rows, cols);
        let (expected, _) = polar.gram_form_corrected_iterations(
            source.clone(),
            rows,
            cols,
            PRODUCTION_ITERATIONS,
            device::GramCorrectionMode::HighPrecision,
        )?;
        let raw = evaluate_mode(
            &polar,
            &source,
            &expected,
            "nvfp4_raw",
            [1.0; MAX_ITERATIONS],
            rows,
            cols,
            PRODUCTION_ITERATIONS,
            device::GramCorrectionMode::Nvfp4GramOnly,
        )?;
        let best_raw = search_best_raw_schedule(
            &polar,
            &source,
            &expected,
            rows,
            cols,
            PRODUCTION_ITERATIONS,
            name,
        )?;
        let best_corrected = search_best_corrected_schedule(
            &polar,
            &source,
            &expected,
            rows,
            cols,
            PRODUCTION_ITERATIONS,
            best_raw.schedule,
        )?;

        println!(
            "nvfp4_gram_form_ratio_search name={name} rows={rows} cols={cols} raw_rel_l2={:.8e} raw_cosine={:.8} best_raw_rel_l2={:.8e} best_raw_cosine={:.8} best_raw_schedule={} best_corrected_rel_l2={:.8e} best_corrected_cosine={:.8} best_corrected_name={} best_corrected_nvfp4_grams={} best_corrected_hi_grams={} best_corrected_rejected={} best_corrected_schedule={}",
            raw.rel_l2,
            raw.cosine,
            best_raw.rel_l2,
            best_raw.cosine,
            schedule_name(&best_raw.schedule),
            best_corrected.rel_l2,
            best_corrected.cosine,
            best_corrected.name,
            best_corrected.nvfp4_grams,
            best_corrected.hi_grams,
            best_corrected.rejected,
            schedule_name(&best_corrected.schedule),
        );
    }

    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_gram_form_production_shapes_report() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
    let f16 = F16TcMatmulModule::from_module(ptx.clone())?;
    let matmul = Nvfp4TcMatmulModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;
    let polar = device::Nvfp4Polar::new(&stream, &f16, &matmul, &quant);

    for (name, rows, cols) in [
        ("attn_c_proj_square", 1024, 1024),
        ("attn_qkv_rect", 1024, 3072),
        ("mlp_up_rect", 1024, 4096),
    ] {
        let source = math::normalized_source(&math::gradient(rows, cols), rows, cols);
        let (expected, expected_stats) = polar.gram_form_corrected_iterations(
            source.clone(),
            rows,
            cols,
            PRODUCTION_ITERATIONS,
            device::GramCorrectionMode::HighPrecision,
        )?;
        println!(
            "nvfp4_gram_form_production_shape name={name} rows={rows} cols={cols} reference=high_precision finite={} hi_grams={} nvfp4_grams={}",
            expected.iter().all(|value| value.is_finite()),
            expected_stats.high_precision_gram_count,
            expected_stats.nvfp4_gram_count,
        );

        for (mode_name, mode) in production_shape_modes() {
            let (actual, stats) = polar.gram_form_corrected_iterations(
                source.clone(),
                rows,
                cols,
                PRODUCTION_ITERATIONS,
                mode,
            )?;
            let finite = actual.iter().all(|value| value.is_finite());
            println!(
                "nvfp4_gram_form_production_shape name={name} mode={mode_name} iterations={PRODUCTION_ITERATIONS} finite={finite} cosine={:.8} rel_l2={:.8e} max_abs={:.8e} nvfp4_grams={} hi_grams={} rejected={}",
                math::cosine(&actual, &expected),
                if finite {
                    math::relative_l2(&actual, &expected)
                } else {
                    f32::INFINITY
                },
                if finite {
                    math::max_abs_error(&actual, &expected)
                } else {
                    f32::INFINITY
                },
                stats.nvfp4_gram_count,
                stats.high_precision_gram_count,
                stats.rejected_stale_steps,
            );
        }
    }

    Ok(())
}
