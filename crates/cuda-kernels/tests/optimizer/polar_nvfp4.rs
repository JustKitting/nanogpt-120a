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
#[path = "polar_nvfp4/mode_cases.rs"]
mod mode_cases;
#[path = "polar_nvfp4/schedule.rs"]
mod schedule;
#[path = "polar_nvfp4/scratch.rs"]
mod scratch;

use mode_cases::{gram_correction_modes, gram_form_correction_modes};
use schedule::{
    evaluate_mode, production_shape_modes, report_schedule_search_best, schedule_name,
    search_best_corrected_schedule, search_best_raw_schedule,
};

const ROWS: usize = 32;
const COLS: usize = 64;
const MAX_ITERATIONS: usize = 8;
const PRODUCTION_ITERATIONS: usize = 5;

fn with_polar<T>(
    run: impl FnOnce(&device::Nvfp4Polar<'_>) -> Result<T, Box<dyn Error>>,
) -> Result<T, Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
    let f16 = F16TcMatmulModule::from_module(ptx.clone())?;
    let matmul = Nvfp4TcMatmulModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;
    let polar = device::Nvfp4Polar::new(&stream, &f16, &matmul, &quant);
    run(&polar)
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_rht_polar_estimator_reports_update_error() -> Result<(), Box<dyn Error>> {
    with_polar(|polar| {
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
    })
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_gram_correction_variants_report_update_error() -> Result<(), Box<dyn Error>> {
    with_polar(|polar| {
        let source = math::normalized_source(&math::gradient(ROWS, COLS), ROWS, COLS);

        for iterations in 1..=MAX_ITERATIONS {
            let expected = math::polar_iterations_f16_leaf(source.clone(), ROWS, COLS, iterations);
            for (name, mode) in gram_correction_modes() {
                let (actual, stats) = polar.gram_corrected_iterations(
                    source.clone(),
                    ROWS,
                    COLS,
                    iterations,
                    mode,
                )?;
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
    })
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_gram_form_correction_variants_report_update_error() -> Result<(), Box<dyn Error>> {
    with_polar(|polar| {
        let source = math::normalized_source(&math::gradient(ROWS, COLS), ROWS, COLS);

        for iterations in 1..=MAX_ITERATIONS {
            let expected =
                math::gram_form_polar_iterations_f16_leaf(source.clone(), ROWS, COLS, iterations);
            for (name, mode) in gram_form_correction_modes() {
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
    })
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_gram_form_safety_schedule_search() -> Result<(), Box<dyn Error>> {
    with_polar(|polar| {
        let source = math::normalized_source(&math::gradient(ROWS, COLS), ROWS, COLS);
        let expected =
            math::gram_form_polar_iterations_f16_leaf(source.clone(), ROWS, COLS, MAX_ITERATIONS);

        let best = search_best_raw_schedule(
            &polar,
            &source,
            &expected,
            ROWS,
            COLS,
            MAX_ITERATIONS,
            "",
            true,
        )?;
        report_schedule_search_best("nvfp4_only", &best);

        let best_corrected = search_best_corrected_schedule(
            &polar,
            &source,
            &expected,
            ROWS,
            COLS,
            MAX_ITERATIONS,
            best.schedule,
            true,
        )?;
        report_schedule_search_best("stale_reject", &best_corrected);

        assert!(
            best.finite,
            "at least one static NVFP4 safety schedule must survive"
        );
        Ok(())
    })
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_gram_form_ratio_schedule_search() -> Result<(), Box<dyn Error>> {
    with_polar(|polar| {
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
                false,
            )?;
            let best_corrected = search_best_corrected_schedule(
                &polar,
                &source,
                &expected,
                rows,
                cols,
                PRODUCTION_ITERATIONS,
                best_raw.schedule,
                false,
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
    })
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_gram_form_production_shapes_report() -> Result<(), Box<dyn Error>> {
    with_polar(|polar| {
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
    })
}
