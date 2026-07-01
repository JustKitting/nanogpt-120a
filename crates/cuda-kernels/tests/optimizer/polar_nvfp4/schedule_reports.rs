use super::schedule::{
    ScheduleEval, evaluate_mode, production_shape_modes, report_schedule_search_best,
    schedule_name, search_best_corrected_schedule, search_best_raw_schedule,
};
use super::{COLS, MAX_ITERATIONS, PRODUCTION_ITERATIONS, ROWS};
use super::{TestResult, device, math, with_polar};

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_gram_form_safety_schedule_search() -> TestResult {
    with_polar(|polar| {
        let source = math::normalized_source(&math::gradient(ROWS, COLS), ROWS, COLS);
        let expected =
            math::gram_form_polar_iterations_f16_leaf(source.clone(), ROWS, COLS, MAX_ITERATIONS);
        let eval = ScheduleEval::new(polar, &source, &expected, ROWS, COLS, MAX_ITERATIONS);

        let best = search_best_raw_schedule(eval, "", true)?;
        report_schedule_search_best("nvfp4_only", &best);

        let best_corrected = search_best_corrected_schedule(eval, best.schedule, true)?;
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
fn nvfp4_gram_form_ratio_schedule_search() -> TestResult {
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
            let eval =
                ScheduleEval::new(polar, &source, &expected, rows, cols, PRODUCTION_ITERATIONS);
            let raw = evaluate_mode(
                eval,
                "nvfp4_raw",
                [1.0; MAX_ITERATIONS],
                device::GramCorrectionMode::Nvfp4GramOnly,
            )?;
            let best_raw = search_best_raw_schedule(eval, name, false)?;
            let best_corrected = search_best_corrected_schedule(eval, best_raw.schedule, false)?;

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
fn nvfp4_gram_form_production_shapes_report() -> TestResult {
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
                let (cosine, rel_l2, max_abs) =
                    math::finite_error_metrics(&actual, &expected, finite);
                println!(
                    "nvfp4_gram_form_production_shape name={name} mode={mode_name} iterations={PRODUCTION_ITERATIONS} finite={finite} cosine={cosine:.8} rel_l2={rel_l2:.8e} max_abs={max_abs:.8e} nvfp4_grams={} hi_grams={} rejected={}",
                    stats.nvfp4_gram_count,
                    stats.high_precision_gram_count,
                    stats.rejected_stale_steps,
                );
            }
        }

        Ok(())
    })
}
