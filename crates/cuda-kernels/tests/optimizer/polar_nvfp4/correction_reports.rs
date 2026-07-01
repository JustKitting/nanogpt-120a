use super::mode_cases::{gram_correction_modes, gram_form_correction_modes};
use super::{COLS, MAX_ITERATIONS, ROWS, TestResult, math, with_polar};

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_gram_correction_variants_report_update_error() -> TestResult {
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
                let (cosine, rel_l2, max_abs) = math::error_metrics(&actual, &expected);
                println!(
                    "nvfp4_gram_correction mode={name} iterations={iterations} finite={finite} cosine={cosine:.8} rel_l2={rel_l2:.8e} max_abs={max_abs:.8e} nvfp4_grams={} hi_grams={} max_defect={:.8e} last_defect={:.8e}",
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
fn nvfp4_gram_form_correction_variants_report_update_error() -> TestResult {
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
                let (cosine, rel_l2, max_abs) = math::error_metrics(&actual, &expected);
                println!(
                    "nvfp4_gram_form_correction mode={name} iterations={iterations} finite={finite} cosine={cosine:.8} rel_l2={rel_l2:.8e} max_abs={max_abs:.8e} nvfp4_grams={} hi_grams={} rejected={} max_defect={:.8e} last_defect={:.8e}",
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
